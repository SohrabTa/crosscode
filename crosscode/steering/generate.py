"""Disulfide steering smoke test: generation pipeline.

Runs Phases 1, 2, 3, and 4a (no-fold metrics) in one go because the
expensive part is loading ProtT5 + the crosscoder; everything else is
cheap per-call.

For each input protein we run:

  baseline-greedy           -- Phase 1: confirms the copy artifact (no steering, no hook).
  identity-hook-greedy      -- Phase 2: encode -> decode + err -> identity round-trip
                               (no z edit). Must produce the same output as baseline.
  steered-greedy            -- Phase 3: z[primary_feature] := c * max_act, decoded back,
                               spliced at the chosen hookpoint, then greedy decode.
  random-feature-greedy     -- Phase 3 control: same z-edit but on a random feature with
                               matched magnitude.

For each generated sequence we compute (Phase 4a):

  identity_to_input         -- copy-break metric
  n_cys                     -- Cys count in output (key disulfide-engineering metric)
  cys_positions             -- positions of any Cys
  delta_n_cys_vs_baseline   -- (n_cys_steered - n_cys_baseline) for the same input

Outputs:

  results_dir/generations.jsonl   -- one record per generation
  results_dir/summary.csv         -- pivoted summary table

Usage::

    uv run python -m crosscode.steering.generate \\
        --checkpoint_dir /path/to/crosscoder/ckpt \\
        --steering_config /path/to/disulfide_steering.json \\
        --inputs_fasta /path/to/inputs.fasta \\
        --results_dir /path/to/results
"""
from __future__ import annotations

import csv
import gc
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput

from crosscode.models.acausal_crosscoder import ModelHookpointAcausalCrosscoder
from crosscode.utils import get_device

PROTT5_MODEL = "Rostlab/prot_t5_xl_uniref50"


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class GenerationRecord:
    input_name: str
    input_seq: str
    config_name: str       # e.g. "baseline-greedy", "steer-f1339-c5", "random-f483-c5"
    output_seq: str
    identity_to_input: float
    n_cys: int
    cys_positions: list[int]
    feature_id: int | None
    boost_c: float | None
    hookpoint: int | None
    wall_seconds: float
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["cys_positions"] = list(d["cys_positions"])
        return d


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _read_fasta(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    name: str | None = None
    seq_chunks: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                entries.append((name, "".join(seq_chunks)))
            name = line[1:].split()[0]
            seq_chunks = []
        elif line.strip():
            seq_chunks.append(line.strip())
    if name is not None:
        entries.append((name, "".join(seq_chunks)))
    return entries


def _spaced(seq: str) -> str:
    return " ".join(re.sub(r"[UZOB]", "X", seq))


def _decode(tokenizer, ids: torch.Tensor) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True).replace(" ", "")


def _identity(a: str, b: str) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if a[i] == b[i]) / n


def _cys_positions(seq: str) -> list[int]:
    return [i for i, c in enumerate(seq) if c == "C"]


# --------------------------------------------------------------------------- #
# Steering session
# --------------------------------------------------------------------------- #


class SteeringSession:
    """Holds ProtT5 (MPS fp16) + crosscoder (CPU fp32) and exposes
    baseline / identity-hook / steered generate methods."""

    def __init__(
        self,
        checkpoint_dir: Path,
        steering_config: dict,
        device: str | None = None,
    ):
        if device is None:
            device = get_device()
        self.device = device
        self.steering_config = steering_config
        self.hookpoint = int(steering_config["best_hookpoint"])
        self.primary_feature = int(steering_config["primary_feature"])
        self.per_feature = steering_config["per_feature"]

        print(f"Loading ProtT5 on {device} (fp16) ...")
        self.tokenizer = T5Tokenizer.from_pretrained(PROTT5_MODEL, do_lower_case=False)
        dtype = torch.float16 if device != "cpu" else torch.float32
        self.model = T5ForConditionalGeneration.from_pretrained(
            PROTT5_MODEL, torch_dtype=dtype
        ).to(device).eval()
        self.encoder = self.model.get_encoder()
        self.n_blocks = len(self.encoder.block)

        print(f"Loading crosscoder on CPU ...")
        self.crosscoder = ModelHookpointAcausalCrosscoder.load(
            checkpoint_dir, device="cpu"
        )
        self.crosscoder.eval()
        assert self.crosscoder.n_hookpoints == self.n_blocks, (
            f"crosscoder.n_hookpoints={self.crosscoder.n_hookpoints} != "
            f"encoder.block count={self.n_blocks}"
        )
        print(
            f"Loaded: n_latents={self.crosscoder.n_latents}, "
            f"n_hookpoints={self.crosscoder.n_hookpoints}, "
            f"d_model={self.crosscoder.d_model}, hookpoint={self.hookpoint}, "
            f"primary_feature={self.primary_feature}"
        )

    # -- Encoder pass with multi-hookpoint capture --------------------------- #

    def _encoder_forward_capture(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor], torch.Tensor]:
        """Run the encoder once, capture all 24 hookpoint activations, and
        return (final_hidden_state, hookpoint_cache, attention_mask_used).

        Each entry in hookpoint_cache is shape [B, S, D] on `device`.
        """
        cache: dict[int, torch.Tensor] = {}

        def make_hook(li: int):
            def hook(_m, _i, output):
                t = output[0] if isinstance(output, tuple) else output
                cache[li] = t.detach()
            return hook

        handles = [
            self.encoder.block[li].register_forward_hook(make_hook(li))
            for li in range(self.n_blocks)
        ]
        try:
            with torch.no_grad():
                out = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
        finally:
            for h in handles:
                h.remove()
        return out.last_hidden_state, cache, attention_mask

    def _encoder_forward_with_splice(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        splice_map: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Re-run the encoder with forward hooks on each block in
        ``splice_map`` that REPLACE the block's output with the supplied
        value. Returns the final encoder hidden state.

        splice_map: {block_idx: replacement_tensor [B, S, D]}
        """
        def make_hook(replacement: torch.Tensor):
            def _hook(_m, _i, output):
                if isinstance(output, tuple):
                    return (replacement,) + output[1:]
                return replacement
            return _hook

        handles = [
            self.encoder.block[block_idx].register_forward_hook(make_hook(rep))
            for block_idx, rep in splice_map.items()
        ]
        try:
            with torch.no_grad():
                out = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
        finally:
            for h in handles:
                h.remove()
        return out.last_hidden_state

    # -- Crosscoder edit ----------------------------------------------------- #

    def _crosscoder_edit_at_hookpoints(
        self,
        hookpoint_cache: dict[int, torch.Tensor],
        edits: list[tuple[int, float, int]],
        seq_len: int,
    ) -> dict[int, torch.Tensor]:
        """Multi-feature, multi-hookpoint version.

        Args:
            hookpoint_cache: {p: [B, S, D]} encoder activations.
            edits: list of (feature_id, boost_value, target_hookpoint). The
                z edit applies all (feature_id, boost_value) pairs to z at
                once; for each unique target_hookpoint we then build a
                spliced replacement tensor.
            seq_len: number of real residues (excluding </s>).

        Returns:
            {block_idx: replacement_tensor [B=1, S, D]} ready to feed back
            into block forward hooks.
        """
        # ---- stack activations ----
        per_hook = [hookpoint_cache[p][:, :seq_len, :].cpu().float()
                    for p in range(self.n_blocks)]
        acts_BSPD = torch.stack(per_hook, dim=2)  # [1, seq_len, P, D]
        flat = acts_BSPD.reshape(-1, self.crosscoder.n_hookpoints, self.crosscoder.d_model)
        x_BMPD = flat.unsqueeze(1)  # [seq_len, 1, P, D]

        with torch.no_grad():
            pre = self.crosscoder.get_pre_bias_BL(x_BMPD)
            if self.crosscoder.b_enc_L is not None:
                pre = pre + self.crosscoder.b_enc_L
            z_orig = self.crosscoder.activation_fn.forward(pre)
            recon_orig = self.crosscoder.decode_BMPD(z_orig)

            if not edits:
                z_edit = z_orig
                recon_edit = recon_orig
            else:
                z_edit = z_orig.clone()
                for fid, boost, _hp in edits:
                    z_edit[:, fid] = boost
                recon_edit = self.crosscoder.decode_BMPD(z_edit)

        # Build replacement per unique target hookpoint.
        target_hps = sorted({hp for _, _, hp in edits}) if edits else []
        out: dict[int, torch.Tensor] = {}
        for L_star in target_hps:
            h_orig_L = acts_BSPD[0, :, L_star, :]
            recon_orig_L = recon_orig[:, 0, L_star, :]
            recon_edit_L = recon_edit[:, 0, L_star, :]
            err_L = h_orig_L - recon_orig_L
            h_new_L = recon_edit_L + err_L

            full_replacement = hookpoint_cache[L_star].clone().to(torch.float32)
            full_replacement[:, :seq_len, :] = h_new_L.to(
                full_replacement.device
            ).unsqueeze(0)
            full_replacement = full_replacement.to(hookpoint_cache[L_star].dtype)
            out[L_star] = full_replacement
        return out

    # -- Generation modes ---------------------------------------------------- #

    def _tokenize(self, seq: str):
        enc = self.tokenizer(
            _spaced(seq),
            add_special_tokens=True, return_tensors="pt", padding=False,
        )
        return (
            enc["input_ids"].to(self.device),
            enc["attention_mask"].to(self.device),
        )

    def _gen_kwargs(self, input_len: int) -> dict:
        return dict(
            max_new_tokens=input_len + 4,
            do_sample=False, num_beams=1,
            return_dict_in_generate=False,
        )

    def _generate_from_encoder_state(
        self,
        encoder_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        input_len: int,
    ) -> torch.Tensor:
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)
        with torch.no_grad():
            out_ids = self.model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                **self._gen_kwargs(input_len),
            )
        return out_ids[0]

    def baseline_greedy(self, seq: str) -> tuple[str, float]:
        """Plain ProtT5 generate(), no hooks, no crosscoder."""
        input_ids, attn = self._tokenize(seq)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids, attention_mask=attn,
                **self._gen_kwargs(len(seq)),
            )
        dt = time.perf_counter() - t0
        return _decode(self.tokenizer, out[0]), dt

    def identity_hook_greedy(self, seq: str) -> tuple[str, float]:
        """Encode -> decode + err round-trip, no z edit. Must equal baseline."""
        input_ids, attn = self._tokenize(seq)
        t0 = time.perf_counter()
        _, cache, attn = self._encoder_forward_capture(input_ids, attn)
        # Empty edit list -> only the identity replacement at self.hookpoint
        replacements = self._crosscoder_edit_at_hookpoints(
            cache, edits=[(0, 0.0, self.hookpoint)], seq_len=len(seq),
        )
        # Overwrite the bogus edit by recomputing with no actual feature change.
        # Cleaner: call again with edits=[] but we still need a target hookpoint.
        # Easiest: pass an edit that boosts a feature to its ORIGINAL value (no-op).
        # We already have z_edit == z_orig path when edits == []; but then
        # we need a target hookpoint to know which replacement to build.
        # Use a dedicated identity-mode helper instead.
        replacements = self._identity_round_trip(cache, len(seq))
        steered_state = self._encoder_forward_with_splice(
            input_ids, attn,
            {hp: rep.to(self.device) for hp, rep in replacements.items()},
        )
        out_ids = self._generate_from_encoder_state(steered_state, attn, len(seq))
        dt = time.perf_counter() - t0
        return _decode(self.tokenizer, out_ids), dt

    def _identity_round_trip(
        self, hookpoint_cache: dict[int, torch.Tensor], seq_len: int
    ) -> dict[int, torch.Tensor]:
        """No-edit round-trip at self.hookpoint -- must reproduce input."""
        per_hook = [hookpoint_cache[p][:, :seq_len, :].cpu().float()
                    for p in range(self.n_blocks)]
        acts_BSPD = torch.stack(per_hook, dim=2)
        flat = acts_BSPD.reshape(-1, self.crosscoder.n_hookpoints, self.crosscoder.d_model)
        x_BMPD = flat.unsqueeze(1)
        with torch.no_grad():
            pre = self.crosscoder.get_pre_bias_BL(x_BMPD)
            if self.crosscoder.b_enc_L is not None:
                pre = pre + self.crosscoder.b_enc_L
            z = self.crosscoder.activation_fn.forward(pre)
            recon = self.crosscoder.decode_BMPD(z)
        L_star = self.hookpoint
        h_orig_L = acts_BSPD[0, :, L_star, :]
        recon_L = recon[:, 0, L_star, :]
        err_L = h_orig_L - recon_L
        h_new_L = recon_L + err_L  # exactly h_orig_L modulo numerical noise
        full = hookpoint_cache[L_star].clone().to(torch.float32)
        full[:, :seq_len, :] = h_new_L.to(full.device).unsqueeze(0)
        full = full.to(hookpoint_cache[L_star].dtype)
        return {L_star: full}

    def steered_greedy(
        self,
        seq: str,
        feature_id: int,
        boost_value: float,
    ) -> tuple[str, float]:
        """Single-feature steering at self.hookpoint."""
        return self.joint_steered_greedy(
            seq, edits=[(feature_id, boost_value, self.hookpoint)]
        )

    def joint_steered_greedy(
        self,
        seq: str,
        edits: list[tuple[int, float, int]],
    ) -> tuple[str, float]:
        """Multi-feature, multi-hookpoint steering.

        edits: list of (feature_id, boost_value, target_hookpoint).
        """
        input_ids, attn = self._tokenize(seq)
        t0 = time.perf_counter()
        _, cache, attn = self._encoder_forward_capture(input_ids, attn)
        replacements = self._crosscoder_edit_at_hookpoints(
            cache, edits=edits, seq_len=len(seq),
        )
        steered_state = self._encoder_forward_with_splice(
            input_ids, attn,
            {hp: rep.to(self.device) for hp, rep in replacements.items()},
        )
        out_ids = self._generate_from_encoder_state(steered_state, attn, len(seq))
        dt = time.perf_counter() - t0
        return _decode(self.tokenizer, out_ids), dt

    # -- Convenience: get the max_activation for a given feature ------------- #

    def max_act(self, feature_id: int) -> float:
        info = self.per_feature.get(str(feature_id))
        if info is None:
            raise KeyError(f"No max_activation for feature {feature_id}")
        return float(info["max_activation"])

    def close(self):
        del self.model, self.crosscoder
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #


def _record(
    input_name: str,
    input_seq: str,
    config_name: str,
    output_seq: str,
    feature_id: int | None,
    boost_c: float | None,
    hookpoint: int | None,
    wall_seconds: float,
    extra: dict | None = None,
    baseline_cys_by_input: dict[str, int] | None = None,
) -> GenerationRecord:
    cys_pos = _cys_positions(output_seq)
    rec = GenerationRecord(
        input_name=input_name,
        input_seq=input_seq,
        config_name=config_name,
        output_seq=output_seq,
        identity_to_input=_identity(input_seq, output_seq),
        n_cys=len(cys_pos),
        cys_positions=cys_pos,
        feature_id=feature_id,
        boost_c=boost_c,
        hookpoint=hookpoint,
        wall_seconds=wall_seconds,
        extra=extra or {},
    )
    if baseline_cys_by_input is not None and input_name in baseline_cys_by_input:
        rec.extra["delta_n_cys_vs_baseline"] = (
            rec.n_cys - baseline_cys_by_input[input_name]
        )
    return rec


def run(
    checkpoint_dir: str | Path,
    steering_config: str | Path,
    inputs_fasta: str | Path,
    results_dir: str | Path,
    boost_c_values: tuple[float, ...] | str = (2.0, 5.0, 10.0),
    cross_features: tuple[int, ...] | str = (1007, 5281, 7299),
    random_features: tuple[int, ...] | str = (4242, 100, 7777),
    skip_random_control: bool = False,
    skip_cross_features: bool = False,
    device: str | None = None,
) -> None:
    """Run the full smoke test."""
    checkpoint_dir = Path(checkpoint_dir)
    steering_config_path = Path(steering_config)
    inputs_fasta = Path(inputs_fasta)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    def _coerce_to_tuple(x, cast):
        if isinstance(x, str):
            return tuple(cast(v) for v in x.split(",") if v.strip())
        if isinstance(x, (int, float)):
            return (cast(x),)
        return tuple(cast(v) for v in x)

    boost_c_values = _coerce_to_tuple(boost_c_values, float)
    cross_features = _coerce_to_tuple(cross_features, int)
    random_features = _coerce_to_tuple(random_features, int)

    with steering_config_path.open() as fh:
        steering_config_dict = json.load(fh)

    inputs = _read_fasta(inputs_fasta)
    print(f"\nLoaded {len(inputs)} input proteins from {inputs_fasta}")
    for name, seq in inputs:
        print(f"  {name:30s} L={len(seq):3d} Cys={seq.count('C')}")

    sess = SteeringSession(checkpoint_dir, steering_config_dict, device=device)
    primary = sess.primary_feature
    primary_max = sess.max_act(primary)
    print(f"\nPrimary feature f/{primary}, max_act={primary_max:.3f}")
    print(f"Boost values c = {list(boost_c_values)} (boost = c * max_act)")
    if not skip_cross_features:
        print(f"Cross-feature controls: f/{list(cross_features)}")
    if not skip_random_control:
        print(f"Random-feature negative controls: f/{list(random_features)}")

    records: list[GenerationRecord] = []
    baseline_n_cys: dict[str, int] = {}

    for name, seq in inputs:
        print(f"\n=== {name} (L={len(seq)}) ===")

        # --- Phase 1: baseline ---
        out_b, dt_b = sess.baseline_greedy(seq)
        rec = _record(name, seq, "baseline-greedy", out_b, None, None, None, dt_b)
        baseline_n_cys[name] = rec.n_cys
        rec.extra["delta_n_cys_vs_baseline"] = 0
        records.append(rec)
        print(
            f"  [baseline]               L={len(out_b):3d}  identity={rec.identity_to_input:.2f}  "
            f"n_Cys={rec.n_cys}  t={dt_b:.1f}s"
        )

        # --- Phase 2: identity hook ---
        out_id, dt_id = sess.identity_hook_greedy(seq)
        rec = _record(
            name, seq, "identity-hook-greedy", out_id, None, None, sess.hookpoint,
            dt_id, baseline_cys_by_input=baseline_n_cys,
        )
        identity_to_baseline = _identity(out_b, out_id)
        rec.extra["identity_to_baseline"] = identity_to_baseline
        records.append(rec)
        print(
            f"  [identity-hook]          L={len(out_id):3d}  "
            f"identity_to_baseline={identity_to_baseline:.2f}  n_Cys={rec.n_cys}  t={dt_id:.1f}s"
        )

        # --- Phase 3: steer primary feature ---
        for c in boost_c_values:
            boost = c * primary_max
            out_s, dt_s = sess.steered_greedy(seq, primary, boost)
            rec = _record(
                name, seq,
                f"steer-f{primary}-c{c}", out_s, primary, c, sess.hookpoint,
                dt_s, baseline_cys_by_input=baseline_n_cys,
            )
            records.append(rec)
            print(
                f"  [steer f/{primary} c={c:>4}]      "
                f"identity={rec.identity_to_input:.2f}  n_Cys={rec.n_cys:>2}  "
                f"ΔCys={rec.extra['delta_n_cys_vs_baseline']:+d}  t={dt_s:.1f}s"
            )

        # --- Phase 3 controls: cross-features ---
        if not skip_cross_features:
            for fid in cross_features:
                fid_max = sess.max_act(fid)
                # use the strongest c for cross-features
                c = max(boost_c_values)
                boost = c * fid_max
                out_x, dt_x = sess.steered_greedy(seq, fid, boost)
                rec = _record(
                    name, seq,
                    f"steer-f{fid}-c{c}", out_x, fid, c, sess.hookpoint,
                    dt_x, baseline_cys_by_input=baseline_n_cys,
                )
                records.append(rec)
                print(
                    f"  [cross f/{fid} c={c:>4}]      "
                    f"identity={rec.identity_to_input:.2f}  n_Cys={rec.n_cys:>2}  "
                    f"ΔCys={rec.extra['delta_n_cys_vs_baseline']:+d}  t={dt_x:.1f}s"
                )

        # --- Phase 3 controls: random features ---
        if not skip_random_control:
            for fid in random_features:
                try:
                    fid_max = sess.max_act(fid)
                except KeyError:
                    # use a fallback: boost = c * primary_max (matched magnitude)
                    fid_max = primary_max
                c = max(boost_c_values)
                boost = c * fid_max
                out_r, dt_r = sess.steered_greedy(seq, fid, boost)
                rec = _record(
                    name, seq,
                    f"random-f{fid}-c{c}", out_r, fid, c, sess.hookpoint,
                    dt_r, baseline_cys_by_input=baseline_n_cys,
                )
                records.append(rec)
                print(
                    f"  [random f/{fid} c={c:>4}]    "
                    f"identity={rec.identity_to_input:.2f}  n_Cys={rec.n_cys:>2}  "
                    f"ΔCys={rec.extra['delta_n_cys_vs_baseline']:+d}  t={dt_r:.1f}s"
                )

    # --- Write outputs ---
    jsonl_path = results_dir / "generations.jsonl"
    with jsonl_path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict()) + "\n")
    print(f"\nWrote {len(records)} records to {jsonl_path}")

    summary_path = results_dir / "summary.csv"
    with summary_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "input_name", "config_name", "feature_id", "boost_c",
            "L_in", "L_out", "identity_to_input", "n_cys",
            "delta_n_cys_vs_baseline", "cys_positions", "wall_seconds",
            "output_seq",
        ])
        for r in records:
            w.writerow([
                r.input_name, r.config_name,
                r.feature_id if r.feature_id is not None else "",
                r.boost_c if r.boost_c is not None else "",
                len(r.input_seq), len(r.output_seq),
                f"{r.identity_to_input:.4f}", r.n_cys,
                r.extra.get("delta_n_cys_vs_baseline", ""),
                ";".join(str(p) for p in r.cys_positions),
                f"{r.wall_seconds:.2f}",
                r.output_seq,
            ])
    print(f"Wrote summary to {summary_path}")

    sess.close()


if __name__ == "__main__":
    import fire

    fire.Fire(run)
