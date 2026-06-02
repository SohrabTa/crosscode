"""Diagnostic: linear probe for secondary structure on crosscoder features.

Answers the question: "Why is per-feature F1 for Helix / Beta strand only ~0.25
when PLMs are known to be excellent at SS prediction?"

If a logistic-regression probe on the full 8192-dim crosscoder latent achieves
high AUROC for Helix/Strand, then the information is *present* in the
crosscoder, just *distributed* across many features (superposition). In that
case, steering should target a learned weight combination, not a single
feature.

If the probe AUROC is also low, then the crosscoder's reconstruction loss is
actually dropping SS-relevant signal, and helix->strand single-feature steering
is unlikely to work.

We baseline against:
 * single best feature for each concept (from the F1 ranking)
 * the raw ProtT5 hidden state at the chosen hookpoint (ceiling for what the
   crosscoder *could* preserve)

Usage::

    uv run python -m crosscode.steering.probe_ss \\
        --checkpoint_dir /path/to/crosscoder/ckpt \\
        --annotations_dir /path/to/eval/processed_annotations \\
        --shard_id 42 \\
        --n_proteins 30 \\
        --max_len 200
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import T5EncoderModel, T5Tokenizer

from crosscode.models.acausal_crosscoder import ModelHookpointAcausalCrosscoder
from crosscode.utils import get_device

PROTT5_MODEL = "Rostlab/prot_t5_xl_uniref50"
HELIX_CONCEPT_NAME = "Helix"
STRAND_CONCEPT_NAME = "Beta strand"
DEFAULT_CONCEPTS: tuple[str, ...] = (HELIX_CONCEPT_NAME, STRAND_CONCEPT_NAME)

# Path to the shortlist produced by ``shortlist_features.py``. Optional —
# only used to compute the "shortlisted features only" AUROC baseline.
# Set the env var ``CROSSCODE_STEERING_SHORTLIST`` to point at the JSON;
# if unset or missing, the shortlisted-features probe baseline is skipped.
import os
_SHORTLIST_ENV = os.environ.get("CROSSCODE_STEERING_SHORTLIST")
DEFAULT_SHORTLIST_PATH = Path(_SHORTLIST_ENV) if _SHORTLIST_ENV else None

# Quiet sklearn's FutureWarnings for cleaner output.
warnings.filterwarnings("ignore", category=FutureWarning)


def _shortlist_subset_auroc(label_name, X_tr_z, X_te_z, y_tr, y_te):
    """If we have a shortlist JSON for this concept, fit an LR on JUST the
    selective features and return the AUROC. Else None.
    """
    if DEFAULT_SHORTLIST_PATH is None or not DEFAULT_SHORTLIST_PATH.exists():
        return None
    try:
        with DEFAULT_SHORTLIST_PATH.open() as fh:
            shortlist = json.load(fh)
        feats = shortlist["feature_ids_selective"].get(label_name, [])
    except Exception:
        return None
    if not feats:
        return None
    sub_tr = X_tr_z[:, feats]
    sub_te = X_te_z[:, feats]
    clf = LogisticRegression(
        C=1.0, max_iter=200, class_weight="balanced", solver="liblinear"
    ).fit(sub_tr, y_tr)
    return roc_auc_score(y_te, clf.decision_function(sub_te))


def _load_prott5(device: str):
    print(f"Loading ProtT5 ({PROTT5_MODEL}) on {device} ...")
    tok = T5Tokenizer.from_pretrained(PROTT5_MODEL, do_lower_case=False)
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = T5EncoderModel.from_pretrained(PROTT5_MODEL, torch_dtype=dtype).to(device).eval()
    return tok, model


def _read_concept_columns(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _pick_proteins(
    shard_dir: Path,
    n_proteins: int,
    max_len: int,
    min_helix: int = 5,
    min_strand: int = 5,
) -> pd.DataFrame:
    """Pick proteins from the shard that have at least some Helix and Strand
    annotations and fit within the length cap."""
    pdat = pd.read_csv(shard_dir / "protein_data.tsv", sep="\t")
    pdat = pdat[pdat["Length"] <= max_len]

    def count_anno(s, marker):
        if not isinstance(s, str):
            return 0
        return s.count(marker)

    pdat["n_helix"] = pdat["Helix"].apply(lambda s: count_anno(s, "HELIX "))
    pdat["n_strand"] = pdat["Beta strand"].apply(lambda s: count_anno(s, "STRAND "))

    # Want a mix: some helix-rich, some strand-rich, some balanced.
    helix_rich = pdat[(pdat["n_helix"] >= min_helix)].nlargest(n_proteins // 2, "n_helix")
    strand_rich = pdat[(pdat["n_strand"] >= min_strand)].nlargest(n_proteins // 2, "n_strand")

    picked = pd.concat([helix_rich, strand_rich]).drop_duplicates(subset=["Entry"]).head(n_proteins)
    print(
        f"Picked {len(picked)} proteins. "
        f"Total residues: {picked['Length'].sum()}. "
        f"Helix-anno: {picked['n_helix'].sum()}, Strand-anno: {picked['n_strand'].sum()}"
    )
    return picked


def _pick_proteins_for_concepts(
    shard_dir: Path,
    concept_cols: list[int],
    n_per_concept: int,
    max_len: int,
    min_positives_per_protein: int = 1,
) -> pd.DataFrame:
    """Pick proteins that have at least ``min_positives_per_protein`` positive
    residues for each of the target concepts, ``n_per_concept`` proteins per
    concept, deduped.

    Uses the precomputed aa_concepts.npz so we don't need to re-parse UniProt
    annotation strings -- works for any concept, including ones not in
    protein_data.tsv (e.g. Domain_Protein kinase, Disulfide bond).
    """
    aa_concepts = sparse.load_npz(shard_dir / "aa_concepts.npz").tocsc()
    aa_meta = pd.read_csv(shard_dir / "aa_metadata.csv")
    pdat = pd.read_csv(shard_dir / "protein_data.tsv", sep="\t")
    pdat = pdat[pdat["Length"] <= max_len][["Entry", "Sequence", "Length"]]

    # Build per-protein positive-count matrix: [n_proteins, n_concepts]
    entry_to_row_block: dict[str, np.ndarray] = {}
    for entry, sub in aa_meta.groupby("Entry"):
        entry_to_row_block[entry] = sub.index.to_numpy()

    chosen_entries: list[str] = []
    seen: set[str] = set()
    for col in concept_cols:
        # number of positive residues per protein for this concept
        col_data = np.asarray(aa_concepts[:, col].todense()).flatten()
        # rank proteins by positive count and take top n_per_concept that fit length cap
        per_protein_pos = []
        valid_entries = set(pdat["Entry"].tolist())
        for entry, rows in entry_to_row_block.items():
            if entry not in valid_entries:
                continue
            n_pos = int((col_data[rows] > 0).sum())
            if n_pos >= min_positives_per_protein:
                per_protein_pos.append((entry, n_pos))
        per_protein_pos.sort(key=lambda x: -x[1])
        added = 0
        for entry, _ in per_protein_pos:
            if entry not in seen:
                chosen_entries.append(entry)
                seen.add(entry)
                added += 1
            if added >= n_per_concept:
                break

    picked = pdat[pdat["Entry"].isin(chosen_entries)].copy()
    print(
        f"Picked {len(picked)} unique proteins across {len(concept_cols)} concepts. "
        f"Total residues: {picked['Length'].sum()}."
    )
    return picked


def _embed_one(
    seq: str,
    tok,
    model,
    device: str,
    n_hookpoints: int = 24,
) -> torch.Tensor:
    """Run ProtT5 encoder, return all-hookpoint activations [L, P, D] for one
    sequence (L = sequence length, P = hookpoints, D = d_model)."""
    spaced = " ".join(re.sub(r"[UZOB]", "X", seq))
    enc = tok(
        spaced, add_special_tokens=True, padding=False, return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    cache: dict[int, torch.Tensor] = {}

    def make_hook(li: int):
        def _hook(_m, _i, out):
            cache[li] = (out[0] if isinstance(out, tuple) else out).detach()
        return _hook

    handles = []
    for li in range(n_hookpoints):
        handles.append(model.encoder.block[li].register_forward_hook(make_hook(li)))
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()

    # Strip </s> -> only keep `actual_len` tokens
    actual_len = len(seq)
    layers = [cache[li][0, :actual_len, :].cpu().float() for li in range(n_hookpoints)]
    return torch.stack(layers, dim=1)  # [L, P, D]


def probe_ss(
    checkpoint_dir: str | Path,
    annotations_dir: str | Path,
    shard_id: int = 42,
    n_proteins: int = 30,
    max_len: int = 200,
    concepts: tuple[str, ...] | str = DEFAULT_CONCEPTS,
    hookpoint_for_raw_baseline: int = 21,
    output_path: str | Path | None = None,
    device: str | None = None,
    random_seed: int = 0,
) -> dict:
    """Run the linear probe diagnostic.

    Args:
        checkpoint_dir: Crosscoder ckpt directory.
        annotations_dir: ``.../processed_annotations`` directory (has
            ``uniprotkb_aa_concepts_columns.txt`` and ``shard_N/...``).
        shard_id: Which shard to use.
        n_proteins: Number of proteins to embed (M1 budget driver).
        max_len: Length cap per protein.
        hookpoint_for_raw_baseline: Which encoder layer to use for the raw
            ProtT5 baseline probe (the "ceiling" the crosscoder is trying to
            preserve). Default 21 = our chosen steering layer.
    """
    if device is None:
        device = get_device()
    annotations_dir = Path(annotations_dir)
    shard_dir = annotations_dir / f"shard_{shard_id}"

    # Normalize concepts argument (fire may pass comma-strings)
    if isinstance(concepts, str):
        concepts = tuple(c.strip() for c in concepts.split(",") if c.strip())
    concepts = tuple(concepts)

    # ---------- inputs ----------
    all_concept_names = _read_concept_columns(
        annotations_dir / "uniprotkb_aa_concepts_columns.txt"
    )
    concept_cols: dict[str, int] = {}
    for c in concepts:
        if c not in all_concept_names:
            raise ValueError(
                f"Concept {c!r} not found in column file. "
                f"First few names: {all_concept_names[:5]}"
            )
        concept_cols[c] = all_concept_names.index(c)
    print("Concept columns:")
    for c, col in concept_cols.items():
        print(f"  {col:>4d}  {c}")

    aa_concepts = sparse.load_npz(shard_dir / "aa_concepts.npz")
    aa_meta = pd.read_csv(shard_dir / "aa_metadata.csv")

    # Per-concept protein selection (the per-concept positive density varies
    # wildly: SS is dense, disulfide / glycosylation are sparse, kinase is
    # rare-but-dense-within-target-proteins). For classic Helix+Strand we
    # keep the original protein_data.tsv-based picker so the earlier results
    # stay reproducible.
    is_classic_ss = set(concepts) == {HELIX_CONCEPT_NAME, STRAND_CONCEPT_NAME}
    if is_classic_ss:
        picked = _pick_proteins(shard_dir, n_proteins, max_len)
    else:
        picked = _pick_proteins_for_concepts(
            shard_dir,
            concept_cols=list(concept_cols.values()),
            n_per_concept=max(1, n_proteins // len(concepts)),
            max_len=max_len,
        )

    seq_by_entry = dict(zip(picked["Entry"], picked["Sequence"]))
    keep = aa_meta["Entry"].isin(seq_by_entry).values
    aa_meta = aa_meta[keep].reset_index(drop=True)
    aa_concepts = aa_concepts[keep]

    # Build per-concept binary labels.
    ys: dict[str, np.ndarray] = {}
    for c, col in concept_cols.items():
        y = (np.asarray(aa_concepts[:, col].todense()).flatten() > 0).astype(np.int8)
        ys[c] = y

    n_res = aa_meta.shape[0]
    print(f"Selected {n_res} residues across {len(seq_by_entry)} proteins.")
    for c, y in ys.items():
        print(f"  {c}: positives = {int(y.sum())} ({100*y.mean():.1f}%)")

    # ---------- ProtT5 + crosscoder ----------
    # Run ProtT5 on the GPU/MPS (it's the expensive part), but keep the
    # crosscoder on CPU so we sidestep MPS dispatch quirks for the
    # multi-dim einsum and dtype issues. The crosscoder forward is
    # cheap per-residue.
    crosscoder = ModelHookpointAcausalCrosscoder.load(
        Path(checkpoint_dir), device="cpu"
    )
    crosscoder.eval()
    tok, prott5 = _load_prott5(device)

    # Per-protein embedding -> concat in the same order as aa_meta.
    z_chunks: list[np.ndarray] = []  # crosscoder latents [Li, 8192]
    raw_chunks: list[np.ndarray] = []  # raw activation at hookpoint_for_raw [Li, D]

    # aa_meta is sorted by (Entry, local_index) within each Entry,
    # so per-Entry contiguous regions match the natural sequence order.
    entries_in_order = aa_meta["Entry"].drop_duplicates().tolist()
    for i, entry in enumerate(entries_in_order):
        seq = seq_by_entry[entry]
        print(f"  [{i+1}/{len(entries_in_order)}] {entry} (L={len(seq)})", flush=True)
        acts_LPD = _embed_one(seq, tok, prott5, device, n_hookpoints=crosscoder.n_hookpoints)
        # _embed_one already returns CPU fp32 tensor. crosscoder lives on CPU.
        x_BMPD = acts_LPD.unsqueeze(1)  # [L, 1, 24, 1024], CPU fp32
        with torch.no_grad():
            pre = crosscoder.get_pre_bias_BL(x_BMPD)
            if crosscoder.b_enc_L is not None:
                pre = pre + crosscoder.b_enc_L
            z_BL = crosscoder.activation_fn.forward(pre)
        z_chunks.append(z_BL.float().numpy())
        raw_chunks.append(
            acts_LPD[:, hookpoint_for_raw_baseline, :].numpy().astype(np.float32)
        )

    X_z = np.concatenate(z_chunks, axis=0).astype(np.float32)
    X_raw = np.concatenate(raw_chunks, axis=0)
    assert X_z.shape[0] == n_res, f"row mismatch: {X_z.shape[0]} vs {n_res}"
    print(f"Feature matrix X_z: {X_z.shape}, X_raw: {X_raw.shape}")

    # ---------- probes ----------
    results = {
        "data": {
            "n_proteins": int(len(entries_in_order)),
            "n_residues": int(n_res),
            "concepts": list(concepts),
            "positive_pct_per_concept": {c: float(100 * ys[c].mean()) for c in concepts},
            "hookpoint_for_raw_baseline": int(hookpoint_for_raw_baseline),
        },
        "probes": {},
    }

    for label_name, y in ys.items():
        print(f"\n--- {label_name} probes ---")
        # Skip if degenerate
        if y.sum() < 20 or (1 - y.mean()) < 0.05:
            print(f"  Skipping {label_name}: too imbalanced (pos={y.sum()})")
            continue

        X_tr_z, X_te_z, X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
            X_z, X_raw, y, test_size=0.3, random_state=random_seed, stratify=y
        )

        # Probe over crosscoder features
        clf_z = LogisticRegression(
            penalty="l2", C=1.0, max_iter=200, class_weight="balanced", solver="liblinear"
        ).fit(X_tr_z, y_tr)
        auroc_z = roc_auc_score(y_te, clf_z.decision_function(X_te_z))

        # Probe over raw ProtT5 hidden state at the steering layer
        clf_raw = LogisticRegression(
            penalty="l2", C=1.0, max_iter=200, class_weight="balanced", solver="liblinear"
        ).fit(X_tr_raw, y_tr)
        auroc_raw = roc_auc_score(y_te, clf_raw.decision_function(X_te_raw))

        # Single-feature baselines: AUROC of each crosscoder feature alone.
        # NaN for features that are constant on the test set (e.g. dead
        # features). Take max over |AUROC - 0.5| so we capture features
        # that are negatively predictive too, but require at least N=10
        # nonzero values in the test set (otherwise AUROC is meaningless).
        single_aurocs = np.full(X_z.shape[1], np.nan, dtype=np.float32)
        min_nonzero_for_scoring = 10
        for j in range(X_z.shape[1]):
            col = X_te_z[:, j]
            if (col != 0).sum() < min_nonzero_for_scoring:
                continue
            if col.max() == col.min():
                continue
            try:
                single_aurocs[j] = roc_auc_score(y_te, col)
            except Exception:
                pass
        # signed distance from 0.5
        valid = ~np.isnan(single_aurocs)
        signed_strength = np.where(valid, np.abs(single_aurocs - 0.5), -1.0)
        best_feat = int(np.argmax(signed_strength))
        best_feat_auroc_raw = float(single_aurocs[best_feat])
        # Report the "effective" AUROC by orienting the feature in the
        # positive direction.
        best_feat_auroc = max(best_feat_auroc_raw, 1.0 - best_feat_auroc_raw)
        n_features_scorable = int(valid.sum())

        # Also: AUROC of just our F1-shortlisted selective features
        # (proxy for: would steering only the shortlisted features even
        # have a chance?). Read shortlist from the canonical path if
        # available.
        sel_auroc = _shortlist_subset_auroc(
            label_name, X_tr_z, X_te_z, y_tr, y_te
        )

        # Top-10 most influential features by |LR coefficient|
        coefs = clf_z.coef_.flatten()
        top10 = np.argsort(-np.abs(coefs))[:10]

        print(f"  AUROC raw ProtT5 (layer {hookpoint_for_raw_baseline}):  {auroc_raw:.3f}")
        print(f"  AUROC crosscoder full (8192-d):           {auroc_z:.3f}")
        if sel_auroc is not None:
            print(f"  AUROC F1-shortlisted selective features:  {sel_auroc:.3f}")
        print(
            f"  AUROC best single crosscoder feature:     {best_feat_auroc:.3f}  "
            f"(feat {best_feat}, raw={best_feat_auroc_raw:.3f}, scorable_features={n_features_scorable})"
        )
        print(f"  Top-10 features by |LR coef|: {top10.tolist()}")

        results["probes"][label_name] = {
            "auroc_raw_prott5": float(auroc_raw),
            "auroc_crosscoder_full": float(auroc_z),
            "auroc_shortlisted_subset": (None if sel_auroc is None else float(sel_auroc)),
            "auroc_single_best_feature": float(best_feat_auroc),
            "single_best_feature_id": best_feat,
            "single_best_feature_auroc_raw": float(best_feat_auroc_raw),
            "n_features_scorable_on_test": n_features_scorable,
            "top10_features_by_abs_coef": top10.tolist(),
            "top10_coefs": [float(coefs[j]) for j in top10],
        }

    if output_path is not None:
        import json
        with open(output_path, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nWrote results to {output_path}")
    return results


if __name__ == "__main__":
    import fire

    fire.Fire(probe_ss)
