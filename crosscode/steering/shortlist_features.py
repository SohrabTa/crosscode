"""Phase 0-A: build the helix/strand feature shortlist from the InterPLM
concept-F1 evaluation output.

For each target secondary-structure concept we pick the union of two top-K
lists:

* top-K by per-amino-acid ``f1`` — features that *specifically* fire on the
  concept;
* top-K by ``recall_per_domain`` — features that fire *somewhere* in most
  occurrences of the concept (broad detectors).

Suppressing/boosting the union covers both selective and broadly-firing
features, which is what we want for a steering smoke test.

The output JSON is consumed by ``pick_hookpoint.py`` (Phase 0-B) and by the
generation scripts.

Usage::

    uv run python -m crosscode.steering.shortlist_features \
        --f1_csv /path/to/test_counts/concept_f1_scores.csv \
        --output_path /path/to/data/steering/steering_features.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

TARGET_CONCEPTS: tuple[str, ...] = ("Helix", "Beta strand")


def _dedupe_by_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-threshold duplicates by keeping the row with max F1
    for each (concept, feature) pair.

    The CSV stores one row per ``threshold_pct`` (0.0, 0.15, 0.5, 0.6, 0.8).
    For most features the rows are identical, but we max-reduce to be safe.
    """
    return (
        df.sort_values("f1", ascending=False)
        .drop_duplicates(subset=["concept", "feature"], keep="first")
        .reset_index(drop=True)
    )


RANKING_METRICS: tuple[str, ...] = ("f1", "f1_per_domain", "recall_per_domain")


def _topk_union(
    df: pd.DataFrame,
    concept: str,
    k: int,
) -> list[dict]:
    """Return the union of top-K features across all RANKING_METRICS for one
    concept. Each entry is annotated with which rankings selected it.
    """
    sub = df[df["concept"] == concept]
    if sub.empty:
        return []

    rankings: dict[str, set[int]] = {
        m: set(sub.nlargest(k, m)["feature"].tolist()) for m in RANKING_METRICS
    }
    union = set().union(*rankings.values())

    sub_union = sub[sub["feature"].isin(union)].copy()
    sub_union["selected_by"] = sub_union["feature"].map(
        lambda f: ",".join(m for m, s in rankings.items() if f in s)
    )
    # Primary sort by f1_per_domain (the InterPLM canonical) descending,
    # then by f1 to break ties.
    sub_union = sub_union.sort_values(
        ["f1_per_domain", "f1"], ascending=False
    )

    return [
        {
            "feature": int(row.feature),
            "f1": float(row.f1),
            "f1_per_domain": float(row.f1_per_domain),
            "precision": float(row.precision),
            "recall": float(row.recall),
            "recall_per_domain": float(row.recall_per_domain),
            "tp": int(row.tp),
            "fp": int(row.fp),
            "selected_by": row.selected_by,
        }
        for row in sub_union.itertuples(index=False)
    ]


def _annotate_selectivity(
    per_concept: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """Tag each feature as ``selective`` (appears in only one concept's
    shortlist) or ``shared`` (appears in multiple).

    Shared features are typically broad "annotated-secondary-structure"
    detectors that don't discriminate helix vs strand, so they are poor
    candidates for differential steering.
    """
    feature_to_concepts: dict[int, set[str]] = {}
    for concept, feats in per_concept.items():
        for f in feats:
            feature_to_concepts.setdefault(f["feature"], set()).add(concept)

    for concept, feats in per_concept.items():
        for f in feats:
            in_concepts = feature_to_concepts[f["feature"]]
            f["selectivity"] = "selective" if in_concepts == {concept} else "shared"
            f["also_in_concepts"] = sorted(in_concepts - {concept})
    return per_concept


def shortlist_features(
    f1_csv: str | Path,
    output_path: str | Path,
    k: int = 5,
    concepts: Iterable[str] = TARGET_CONCEPTS,
) -> dict:
    """Build the steering feature shortlist and write it to ``output_path``.

    Args:
        f1_csv: Path to ``concept_f1_scores.csv`` produced by the InterPLM
            eval pipeline (typically inside ``test_counts/`` for the
            test-set ranking).
        output_path: Where to write the JSON shortlist.
        k: Top-K to take from each ranking. The output per concept may
            contain up to ``2*k`` features after the union.
        concepts: Concepts to shortlist. Defaults to ``("Helix",
            "Beta strand")``.

    Returns:
        The shortlist dict (also written to ``output_path``).
    """
    f1_csv = Path(f1_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(f1_csv)
    df = df[df["concept"].isin(list(concepts))]
    df = _dedupe_by_feature(df)

    per_concept = {concept: _topk_union(df, concept, k) for concept in concepts}
    per_concept = _annotate_selectivity(per_concept)

    selective_ids = {
        c: sorted({f["feature"] for f in v if f["selectivity"] == "selective"})
        for c, v in per_concept.items()
    }
    all_ids = {c: sorted({f["feature"] for f in v}) for c, v in per_concept.items()}

    shortlist = {
        "metadata": {
            "f1_csv": str(f1_csv),
            "k_per_ranking": k,
            "ranking_metrics": list(RANKING_METRICS),
            "concepts": list(concepts),
            "per_concept_counts": {c: len(v) for c, v in per_concept.items()},
            "per_concept_selective_counts": {c: len(v) for c, v in selective_ids.items()},
        },
        "features": per_concept,
        # All shortlisted features (selective + shared), per concept.
        "feature_ids_all": all_ids,
        # Recommended for differential steering: only concept-selective features.
        "feature_ids_selective": selective_ids,
    }

    with output_path.open("w") as fh:
        json.dump(shortlist, fh, indent=2)

    _print_summary(shortlist)
    return shortlist


def _print_summary(shortlist: dict) -> None:
    print(f"Read F1 CSV: {shortlist['metadata']['f1_csv']}")
    for concept, feats in shortlist["features"].items():
        n_sel = sum(1 for f in feats if f["selectivity"] == "selective")
        n_shr = sum(1 for f in feats if f["selectivity"] == "shared")
        print(f"\n=== {concept} ({len(feats)} features: {n_sel} selective, {n_shr} shared) ===")
        print(
            f"{'feature':>8}  {'f1':>6}  {'f1/dom':>6}  {'precision':>9}  "
            f"{'recall':>6}  {'recall/dom':>10}  {'sel':<9}  {'selected_by'}"
        )
        for f in feats:
            print(
                f"{f['feature']:>8d}  {f['f1']:>6.3f}  {f['f1_per_domain']:>6.3f}  "
                f"{f['precision']:>9.3f}  {f['recall']:>6.3f}  "
                f"{f['recall_per_domain']:>10.3f}  {f['selectivity']:<9}  "
                f"{f['selected_by']}"
            )
    print(f"\nSelective feature_ids (recommended for differential steering):")
    for c, ids in shortlist["feature_ids_selective"].items():
        print(f"  {c}: {ids}")


if __name__ == "__main__":
    import fire

    fire.Fire(shortlist_features)
