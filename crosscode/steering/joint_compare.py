"""Compare three steering regimes at a fixed boost magnitude:

  A. f/3285 alone @ L=3   (Cys identity)
  B. f/1339 alone @ L=21  (disulfide context)
  C. f/3285 @ L=3 + f/1339 @ L=21 jointly

Question: does adding disulfide-context steering on top of Cys-identity
steering bias the inserted Cys toward positions that could form a disulfide
in the fold? (We measure Cys positions here; folding is the next step.)

Usage::

    uv run python -m crosscode.steering.joint_compare \\
        --checkpoint_dir /path/to/crosscoder/ckpt \\
        --inputs_fasta /path/to/inputs.fasta \\
        --results_dir /path/to/results \\
        --boost_c 2.0
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from crosscode.steering.generate import (
    SteeringSession,
    _identity,
    _cys_positions,
    _read_fasta,
)


# Hardcoded for this experiment (could be parameterized later).
FEATURE_3285_MAX_ACT = 31.018
FEATURE_3285_HOOKPOINT = 3
FEATURE_1339_MAX_ACT = 27.669
FEATURE_1339_HOOKPOINT = 21


def run(
    checkpoint_dir: str | Path,
    inputs_fasta: str | Path,
    results_dir: str | Path,
    boost_c: float = 2.0,
    device: str | None = None,
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    inputs_fasta = Path(inputs_fasta)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    inputs = _read_fasta(inputs_fasta)
    print(f"Loaded {len(inputs)} inputs")
    for n, s in inputs:
        print(f"  {n:30s} L={len(s):3d}")

    # Build a minimal steering_config dict for the session.
    steering_config_dict = {
        "concept": "joint-cys-and-disulfide",
        "primary_feature": 3285,
        "best_hookpoint": FEATURE_3285_HOOKPOINT,  # not really used in joint mode
        "per_feature": {
            "3285": {"max_activation": FEATURE_3285_MAX_ACT},
            "1339": {"max_activation": FEATURE_1339_MAX_ACT},
        },
    }
    sess = SteeringSession(checkpoint_dir, steering_config_dict, device=device)

    print(f"\nBoost magnitude c = {boost_c}")
    print(f"  -> f/3285 boost = {boost_c * FEATURE_3285_MAX_ACT:.2f}")
    print(f"  -> f/1339 boost = {boost_c * FEATURE_1339_MAX_ACT:.2f}")

    records: list[dict] = []
    for name, seq in inputs:
        print(f"\n=== {name} (L={len(seq)}) ===")

        # baseline
        out_b, dt_b = sess.baseline_greedy(seq)
        baseline_n_cys = out_b.count("C")
        rec = {
            "input_name": name, "input_seq": seq,
            "config": "baseline-greedy",
            "boost_c": None, "edits": [],
            "output_seq": out_b,
            "identity_to_input": _identity(seq, out_b),
            "n_cys": baseline_n_cys,
            "cys_positions": _cys_positions(out_b),
            "wall_seconds": dt_b,
        }
        records.append(rec)
        print(f"  [baseline]                   id={rec['identity_to_input']:.2f}  n_Cys={rec['n_cys']}  t={dt_b:.1f}s")

        configs = [
            ("solo-f3285", [(3285, boost_c * FEATURE_3285_MAX_ACT, FEATURE_3285_HOOKPOINT)]),
            ("solo-f1339", [(1339, boost_c * FEATURE_1339_MAX_ACT, FEATURE_1339_HOOKPOINT)]),
            ("joint-f3285-f1339", [
                (3285, boost_c * FEATURE_3285_MAX_ACT, FEATURE_3285_HOOKPOINT),
                (1339, boost_c * FEATURE_1339_MAX_ACT, FEATURE_1339_HOOKPOINT),
            ]),
        ]
        for cfg_name, edits in configs:
            out_s, dt_s = sess.joint_steered_greedy(seq, edits)
            cys_pos = _cys_positions(out_s)
            rec = {
                "input_name": name, "input_seq": seq,
                "config": cfg_name,
                "boost_c": boost_c, "edits": edits,
                "output_seq": out_s,
                "identity_to_input": _identity(seq, out_s),
                "n_cys": len(cys_pos),
                "cys_positions": cys_pos,
                "delta_n_cys_vs_baseline": len(cys_pos) - baseline_n_cys,
                "wall_seconds": dt_s,
            }
            records.append(rec)
            print(
                f"  [{cfg_name:>20s}]  id={rec['identity_to_input']:.2f}  "
                f"n_Cys={rec['n_cys']:>2}  positions={cys_pos}  t={dt_s:.1f}s"
            )

    jsonl_path = results_dir / "joint_compare.jsonl"
    with jsonl_path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(records)} records to {jsonl_path}")

    csv_path = results_dir / "joint_compare.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["input_name", "config", "L_in", "L_out", "identity_to_input",
                    "n_cys", "cys_positions", "output_seq"])
        for r in records:
            w.writerow([
                r["input_name"], r["config"], len(r["input_seq"]),
                len(r["output_seq"]), f"{r['identity_to_input']:.4f}",
                r["n_cys"], ";".join(str(p) for p in r["cys_positions"]),
                r["output_seq"],
            ])
    print(f"Wrote summary to {csv_path}")
    sess.close()


if __name__ == "__main__":
    import fire

    fire.Fire(run)
