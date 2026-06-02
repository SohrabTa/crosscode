"""Phase 0-B: pick the intervention hookpoint and capture per-feature max
activations.

For an acausal crosscoder, a single latent ``k`` has 24 decoder vectors —
one per encoder hookpoint. The L2 norm of ``W_dec[k, 0, p, :]`` measures
how much latent ``k`` *writes* to the activation at hookpoint ``p`` during
reconstruction. Intervening on ``z[k]`` at hookpoint ``p`` therefore has
its largest direct effect at the hookpoint with the largest decoder norm
for ``k``.

For *differential* steering (helix vs strand) we want a single hookpoint
where both feature groups have strong representation, so the score
combines them with a geometric-mean-like rule that punishes asymmetry.

We also load ``max_activations_per_feature.pt`` (saved by the InterPLM
normalization step) and write the per-feature max-activation values into
the JSON shortlist. These are the natural reference for the
``boost = c * max_act`` steering magnitude.

Usage::

    uv run python -m crosscode.steering.pick_hookpoint \
        --checkpoint_dir /path/to/crashed_epoch_0_step_2519836 \
        --shortlist_path /path/to/steering_features.json
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from crosscode.models.acausal_crosscoder import ModelHookpointAcausalCrosscoder
from crosscode.utils import get_device


def _decoder_norms(
    crosscoder, feature_ids: list[int], unfold_scale: bool = True
) -> torch.Tensor:
    """Return ``[n_features, n_hookpoints]`` L2 norms of decoder vectors.

    If ``unfold_scale`` is True and the crosscoder has folded per-hookpoint
    scaling factors, multiply each hookpoint's norm by its
    ``folded_scaling_factors_out_Xo`` value. This puts the norms back into
    *normalized activation space* so hookpoints with naturally larger raw
    activations (later T5 layers) don't dominate the ranking.

    This matches the per-feature "Cross-Layer Representation" plot in the
    InterPLM dashboard (``interplm/dashboard/app.py::_plot_crosscoder_layer_norms``),
    which does ``w_dec * folded_scaling_factors_out_Xo`` before computing
    the L2 norm to show "where the feature is most strongly represented
    in the unscaled residual stream".
    """
    W_dec = crosscoder.W_dec_LMPD  # [n_latents, n_models, n_hookpoints, d_model]
    assert W_dec.shape[1] == 1, "expected n_models == 1 for ProtT5 setup"
    sub = W_dec[feature_ids, 0, :, :]  # [F, P, D]
    norms = sub.norm(dim=-1).detach().cpu()  # [F, P]

    if unfold_scale and bool(crosscoder.is_folded.item()):
        scale_out_P = crosscoder.folded_scaling_factors_out_Xo[0].detach().cpu()  # [P]
        norms = norms * scale_out_P[None, :]
    return norms


def _score_hookpoint(
    helix_norms: torch.Tensor, strand_norms: torch.Tensor
) -> torch.Tensor:
    """Combined per-hookpoint score that rewards balanced helix+strand
    representation.

    ``helix_norms``, ``strand_norms``: ``[n_features_concept, n_hookpoints]``.

    Score per hookpoint ``p``: ``sqrt(sum(helix_norms[:, p]^2) *
    sum(strand_norms[:, p]^2))``. Geometric mean of the per-concept summed
    squared norms — a hookpoint that is strong for only one concept gets
    penalised vs one that is strong for both.
    """
    helix_sq = (helix_norms**2).sum(dim=0)
    strand_sq = (strand_norms**2).sum(dim=0)
    return (helix_sq * strand_sq).sqrt()


def pick_hookpoint(
    checkpoint_dir: str | Path,
    shortlist_path: str | Path,
    max_acts_path: str | Path | None = None,
    device: str | None = None,
) -> dict:
    """Load the crosscoder, score each hookpoint, and update the shortlist
    JSON in-place with the chosen hookpoint and per-feature max activations.

    Args:
        checkpoint_dir: Directory containing the standard SaveableModule
            files (``model_cfg.yaml`` + ``model.pt``) plus
            ``max_activations_per_feature.pt``.
        shortlist_path: JSON produced by ``shortlist_features.py``.
        max_acts_filename: File holding the per-feature max activation
            tensor (saved by InterPLM's ``normalize.py``).
        device: torch device. ``None`` lets ``get_device()`` decide.

    Returns:
        The updated shortlist dict (also written to disk).
    """
    checkpoint_dir = Path(checkpoint_dir)
    shortlist_path = Path(shortlist_path)

    with shortlist_path.open() as fh:
        shortlist = json.load(fh)

    if device is None:
        device = get_device()
    print(f"Loading crosscoder from {checkpoint_dir} (device={device}) ...")
    crosscoder = ModelHookpointAcausalCrosscoder.load(checkpoint_dir, device=device)
    crosscoder.eval()
    n_latents = crosscoder.n_latents
    n_hookpoints = crosscoder.n_hookpoints
    print(
        f"Loaded crosscoder: n_latents={n_latents}, "
        f"n_hookpoints={n_hookpoints}, d_model={crosscoder.d_model}"
    )

    # Hookpoint scoring uses only the *selective* features per concept —
    # shared features don't discriminate so they shouldn't bias the choice.
    selective = shortlist["feature_ids_selective"]
    helix_ids = selective["Helix"]
    strand_ids = selective["Beta strand"]

    helix_norms = _decoder_norms(crosscoder, helix_ids, unfold_scale=True)
    strand_norms = _decoder_norms(crosscoder, strand_ids, unfold_scale=True)
    helix_norms_folded = _decoder_norms(crosscoder, helix_ids, unfold_scale=False)
    strand_norms_folded = _decoder_norms(crosscoder, strand_ids, unfold_scale=False)

    score = _score_hookpoint(helix_norms, strand_norms)  # [P]
    best_hookpoint = int(score.argmax().item())

    # Decoder norms for *all* shortlisted features (selective + shared)
    # so per-feature info is complete in the JSON.
    all_ids = sorted({fid for v in shortlist["features"].values() for f in v
                      for fid in [f["feature"]]})
    all_norms = _decoder_norms(crosscoder, all_ids, unfold_scale=True)

    # Also load max activations for boost magnitudes.
    if max_acts_path is None:
        max_acts_path = _find_max_acts(checkpoint_dir)
    max_acts_path = Path(max_acts_path)
    print(f"Loading max activations from {max_acts_path} ...")
    max_acts_blob = torch.load(max_acts_path, map_location="cpu", weights_only=False)
    max_acts = _extract_max_acts(max_acts_blob, n_latents)
    print(f"max_acts shape: {tuple(max_acts.shape)}, dtype: {max_acts.dtype}")

    per_feature_info = _build_per_feature_info(
        all_ids, all_norms, max_acts, best_hookpoint,
    )

    shortlist["hookpoint_selection"] = {
        "best_hookpoint": best_hookpoint,
        "score_per_hookpoint": [float(x) for x in score.tolist()],
        "helix_norm_per_hookpoint_unfolded": [
            float(x) for x in (helix_norms**2).sum(dim=0).sqrt().tolist()
        ],
        "strand_norm_per_hookpoint_unfolded": [
            float(x) for x in (strand_norms**2).sum(dim=0).sqrt().tolist()
        ],
        "helix_norm_per_hookpoint_folded": [
            float(x) for x in (helix_norms_folded**2).sum(dim=0).sqrt().tolist()
        ],
        "strand_norm_per_hookpoint_folded": [
            float(x) for x in (strand_norms_folded**2).sum(dim=0).sqrt().tolist()
        ],
        "scoring_method": (
            "geomean(sum_h ||W_dec_unfolded[h,0,p,:]||^2, sum_s ||W_dec_unfolded[s,0,p,:]||^2); "
            "'unfolded' = folded W_dec multiplied by folded_scaling_factors_out_Xo[p] "
            "to compensate for T5's per-layer activation magnitude growth"
        ),
        "checkpoint_dir": str(checkpoint_dir),
    }
    shortlist["per_feature_max_activation"] = per_feature_info

    with shortlist_path.open("w") as fh:
        json.dump(shortlist, fh, indent=2)

    _print_summary(score, helix_norms, strand_norms, best_hookpoint, per_feature_info)
    return shortlist


def _find_max_acts(checkpoint_dir: Path) -> Path:
    """Locate ``max_activations_per_feature.pt`` in the checkpoint dir or
    any immediate eval subdir. Prefer the 67k eval if multiple match.
    """
    candidates = sorted(checkpoint_dir.rglob("max_activations_per_feature.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No max_activations_per_feature.pt found under {checkpoint_dir}"
        )
    # Prefer the 67k eval if available, else the first hit.
    for c in candidates:
        if "67k" in str(c):
            return c
    return candidates[0]


def _extract_max_acts(blob, n_latents: int) -> torch.Tensor:
    """``max_activations_per_feature.pt`` may be saved as a bare tensor or
    as a dict — handle both."""
    if isinstance(blob, torch.Tensor):
        t = blob
    elif isinstance(blob, dict):
        # Common keys we might find.
        for k in ("max_activations", "max_acts", "max", "value"):
            if k in blob:
                t = blob[k]
                break
        else:
            raise ValueError(
                f"Couldn't find max activations in dict; keys={list(blob.keys())}"
            )
    else:
        raise TypeError(f"Unexpected blob type: {type(blob)}")

    if t.dim() == 1:
        assert t.shape[0] == n_latents, (
            f"Expected {n_latents} latents, got max-acts shape {tuple(t.shape)}"
        )
        return t
    # Possible shape [n_models, n_hookpoints, n_latents] or similar.
    flat = t.reshape(-1, n_latents) if t.shape[-1] == n_latents else t.reshape(n_latents, -1).T
    return flat.max(dim=0).values


def _build_per_feature_info(
    all_ids: list[int],
    all_norms: torch.Tensor,
    max_acts: torch.Tensor,
    best_hookpoint: int,
) -> dict:
    """Map each shortlisted feature -> {max_act, decoder norms per
    hookpoint, argmax hookpoint, norm at chosen hookpoint}."""
    out: dict[str, dict] = {}
    for i, fid in enumerate(all_ids):
        row = all_norms[i]
        out[str(fid)] = {
            "max_activation": float(max_acts[fid].item()),
            "decoder_norm_at_best_hookpoint": float(row[best_hookpoint].item()),
            "argmax_hookpoint": int(row.argmax().item()),
            "decoder_norm_per_hookpoint": [float(x) for x in row.tolist()],
        }
    return out


def _print_summary(score, helix_norms, strand_norms, best_hp, per_feature_info):
    print("\n=== Per-hookpoint scale-invariant decoder norms ===")
    print("(helix/strand_norm here = sqrt(sum ||W_dec_unfolded[k,0,p,:]||^2 over selective features))")
    helix_sum = (helix_norms**2).sum(dim=0).sqrt()
    strand_sum = (strand_norms**2).sum(dim=0).sqrt()
    print(f"{'L':>3}  {'helix_norm':>10}  {'strand_norm':>11}  {'score':>10}")
    for p in range(score.shape[0]):
        marker = " <-- best" if p == best_hp else ""
        print(
            f"{p:>3d}  {helix_sum[p].item():>10.4f}  "
            f"{strand_sum[p].item():>11.4f}  {score[p].item():>10.4f}{marker}"
        )
    print(
        f"\nChosen intervention hookpoint: L = {best_hp} (0-indexed)"
        f"  ==  Layer {best_hp + 1} in the InterPLM dashboard's 1-indexed convention"
    )

    print("\n=== Per-feature info at best hookpoint ===")
    print(f"{'feature':>8}  {'max_act':>9}  {'norm@best':>10}  {'argmax_HP':>9}")
    for fid, info in sorted(per_feature_info.items(), key=lambda kv: int(kv[0])):
        print(
            f"{int(fid):>8d}  {info['max_activation']:>9.3f}  "
            f"{info['decoder_norm_at_best_hookpoint']:>10.4f}  "
            f"{info['argmax_hookpoint']:>9d}"
        )


if __name__ == "__main__":
    import fire

    fire.Fire(pick_hookpoint)
