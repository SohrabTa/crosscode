"""Disulfide-steering setup: pick the intervention hookpoint and capture max
activations for the disulfide-bond features identified by the InterPLM eval.

Writes ``data/steering/disulfide_steering.json`` with:
  * the chosen intervention hookpoint (0-indexed; +1 for dashboard convention)
  * per-feature decoder norm distribution across all 24 hookpoints
  * per-feature max activation (boost reference for ``c * max_act``)

The default feature set is::

    {1339, 1007, 5281, 7299}

ranked by F1_per_domain from the 67k eval. f/1339 (F1=0.76) is the primary
steering target; the other three are redundant cross-validation features.

Usage::

    uv run python -m crosscode.steering.disulfide_setup \\
        --checkpoint_dir /path/to/crosscoder/ckpt \\
        --output_path /path/to/data/steering/disulfide_steering.json
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from crosscode.models.acausal_crosscoder import ModelHookpointAcausalCrosscoder
from crosscode.steering.pick_hookpoint import (
    _decoder_norms,
    _extract_max_acts,
    _find_max_acts,
)
from crosscode.utils import get_device

# F1_per_domain ranked features for "Disulfide bond" from
# data/crosscoder_eval/pre-auxfix/real/uniprotkb_modern_score45_67k/test_counts/heldout_top_pairings.csv
# and the dashboard's Sig_concepts_per_feature.csv.
DEFAULT_DISULFIDE_FEATURES: tuple[int, ...] = (1339, 1007, 5281, 7299)


def disulfide_setup(
    checkpoint_dir: str | Path,
    output_path: str | Path,
    feature_ids: tuple[int, ...] | str = DEFAULT_DISULFIDE_FEATURES,
    primary_feature: int = 1339,
    max_acts_path: str | Path | None = None,
    device: str | None = None,
) -> dict:
    """Pick the hookpoint at which ``primary_feature`` is most strongly
    represented (in scale-invariant decoder norm) and write a JSON config
    for the steering pipeline.

    Args:
        checkpoint_dir: SaveableModule directory (``model.pt`` +
            ``model_cfg.yaml``).
        output_path: JSON to write.
        feature_ids: Disulfide-associated features to include. The primary
            feature drives hookpoint selection; the others come along for
            cross-validation.
        primary_feature: The feature whose argmax-hookpoint defines the
            intervention layer. Default 1339 (F1=0.76).
        max_acts_path: Optional path to ``max_activations_per_feature.pt``.
            If omitted, search inside ``checkpoint_dir`` (preferring the
            67k eval subfolder).
        device: torch device; defaults to ``get_device()``.
    """
    checkpoint_dir = Path(checkpoint_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(feature_ids, str):
        feature_ids = tuple(int(x.strip()) for x in feature_ids.split(",") if x.strip())
    feature_ids = tuple(feature_ids)
    assert primary_feature in feature_ids, (
        f"primary_feature {primary_feature} must be in feature_ids {feature_ids}"
    )

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

    # Scale-invariant decoder norms (matches dashboard convention).
    norms = _decoder_norms(crosscoder, list(feature_ids), unfold_scale=True)  # [F, P]

    # Hookpoint pick: argmax of the primary feature's norm.
    primary_idx_in_set = feature_ids.index(primary_feature)
    primary_norms = norms[primary_idx_in_set]  # [P]
    best_hookpoint = int(primary_norms.argmax().item())

    # Max activations for boost magnitudes.
    if max_acts_path is None:
        max_acts_path = _find_max_acts(checkpoint_dir)
    max_acts_path = Path(max_acts_path)
    print(f"Loading max activations from {max_acts_path} ...")
    max_acts_blob = torch.load(max_acts_path, map_location="cpu", weights_only=False)
    max_acts = _extract_max_acts(max_acts_blob, n_latents)

    # Per-feature info dict.
    per_feature: dict[str, dict] = {}
    for i, fid in enumerate(feature_ids):
        row = norms[i]
        per_feature[str(fid)] = {
            "max_activation": float(max_acts[fid].item()),
            "argmax_hookpoint": int(row.argmax().item()),
            "decoder_norm_at_best_hookpoint": float(row[best_hookpoint].item()),
            "decoder_norm_per_hookpoint": [float(x) for x in row.tolist()],
            "is_primary": (fid == primary_feature),
        }

    result = {
        "concept": "Disulfide bond",
        "primary_feature": int(primary_feature),
        "all_features": [int(x) for x in feature_ids],
        "best_hookpoint": best_hookpoint,
        "best_hookpoint_dashboard_layer": best_hookpoint + 1,
        "scoring_method": (
            "argmax over hookpoints of scale-invariant decoder norm "
            "||W_dec[primary_feature, 0, p, :]|| * folded_scaling_factors_out_Xo[p]"
        ),
        "checkpoint_dir": str(checkpoint_dir),
        "max_acts_path": str(max_acts_path),
        "per_feature": per_feature,
    }

    output_path.write_text(json.dumps(result, indent=2))
    _print_summary(result)
    return result


def _print_summary(result: dict) -> None:
    print()
    print(f"Primary feature: f/{result['primary_feature']}")
    print(
        f"Chosen intervention hookpoint: L = {result['best_hookpoint']} "
        f"(0-indexed)  ==  Layer {result['best_hookpoint_dashboard_layer']} "
        f"(dashboard 1-indexed)"
    )
    print()
    print(f"{'feature':>8}  {'max_act':>9}  {'argmax_HP':>9}  {'norm@best':>10}  primary")
    for fid_str, info in result["per_feature"].items():
        print(
            f"{int(fid_str):>8d}  {info['max_activation']:>9.3f}  "
            f"{info['argmax_hookpoint']:>9d}  "
            f"{info['decoder_norm_at_best_hookpoint']:>10.4f}  "
            f"{'<-- primary' if info['is_primary'] else ''}"
        )
    print()
    # Per-hookpoint norms for the primary feature
    primary = next(iter(v for v in result["per_feature"].values() if v["is_primary"]))
    print(f"Primary feature decoder-norm-per-hookpoint (scale-invariant):")
    print(f"{'L':>3}  {'norm':>8}")
    for p, n in enumerate(primary["decoder_norm_per_hookpoint"]):
        marker = " <-- best" if p == result["best_hookpoint"] else ""
        print(f"{p:>3d}  {n:>8.4f}{marker}")


if __name__ == "__main__":
    import fire

    fire.Fire(disulfide_setup)
