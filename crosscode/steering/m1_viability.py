"""Phase 0.5 -- M1 viability check.

Goal: confirm we can actually run ProtT5-XL (T5ForConditionalGeneration)
generation + a single crosscoder forward pass on the laptop in a tolerable
time budget. Go/no-go gate before writing the full steering pipeline.

What we measure:
 1. Load time + peak memory for ProtT5 (fp16 on MPS).
 2. Load time for the crosscoder (fp32 on CPU -- we already know MPS
    dispatch is flaky for the big multi-dim einsum so we keep it on CPU).
 3. Wall time for ONE greedy generate() call on a small protein.
 4. Whether ProtT5's greedy decode reproduces the input (the colleague's
    decoder-copy artifact). If it does NOT copy, that's also useful info.
 5. Wall time for one crosscoder encode pass (needed for the steering
    hook in Phase 2).

Usage::

    uv run python -m crosscode.steering.m1_viability \\
        --checkpoint_dir /path/to/crosscoder/ckpt
"""
from __future__ import annotations

import gc
import re
import time
from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from crosscode.models.acausal_crosscoder import ModelHookpointAcausalCrosscoder
from crosscode.utils import get_device

PROTT5_MODEL = "Rostlab/prot_t5_xl_uniref50"

# Smallest verified Cys-free smoke-test input.
SMOKE_INPUT_NAME = "engrailed_1ENH"
SMOKE_INPUT_SEQ = (
    "EKRPRTAFSSEQLARLKREFNENRYLTERRRQQLSSELGLNEAQIKIWFQNKRAKI"
)  # L=56


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def _mps_mem_alloc() -> int | None:
    if torch.backends.mps.is_available():
        try:
            return torch.mps.current_allocated_memory()
        except Exception:
            return None
    return None


def _spaced(seq: str) -> str:
    return " ".join(re.sub(r"[UZOB]", "X", seq))


def _gen_kwargs(input_len: int) -> dict:
    """Standard greedy-decode kwargs; cap output at input_len + 4 to avoid
    runaway sequences during the smoke test."""
    return dict(
        max_new_tokens=input_len + 4,
        do_sample=False,
        num_beams=1,
        return_dict_in_generate=False,
    )


def _decode_protT5_output(tokenizer, output_ids: torch.Tensor) -> str:
    """Convert ProtT5 decoder output token ids back to an AA sequence."""
    # T5 decoder starts with pad token; greedy generate prepends it.
    # tokenizer.decode strips special tokens; output uses single-letter AA.
    out = tokenizer.decode(output_ids, skip_special_tokens=True)
    return out.replace(" ", "")


def m1_viability(
    checkpoint_dir: str | Path,
    n_warm: int = 1,
    n_timed: int = 1,
) -> dict:
    """Run the M1 viability check end-to-end."""
    print("=" * 60)
    print("Phase 0.5 -- M1 viability check")
    print("=" * 60)
    device = get_device()
    print(f"Device: {device}")
    print(f"Test input: {SMOKE_INPUT_NAME} (L={len(SMOKE_INPUT_SEQ)})")
    print()

    # --- ProtT5 load ---
    print("[1/4] Loading ProtT5-XL (T5ForConditionalGeneration) ...")
    t0 = time.perf_counter()
    tokenizer = T5Tokenizer.from_pretrained(PROTT5_MODEL, do_lower_case=False)
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = T5ForConditionalGeneration.from_pretrained(
        PROTT5_MODEL, torch_dtype=dtype
    ).to(device).eval()
    t_prott5_load = time.perf_counter() - t0
    mem_after_prott5 = _mps_mem_alloc()
    print(
        f"   loaded in {t_prott5_load:.1f}s; dtype={dtype}; "
        f"MPS alloc={_fmt_bytes(mem_after_prott5) if mem_after_prott5 is not None else 'n/a'}"
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   total params: {n_params/1e9:.2f}B")
    print()

    # --- Crosscoder load (CPU) ---
    print("[2/4] Loading crosscoder (CPU, fp32) ...")
    t0 = time.perf_counter()
    crosscoder = ModelHookpointAcausalCrosscoder.load(Path(checkpoint_dir), device="cpu")
    crosscoder.eval()
    t_cc_load = time.perf_counter() - t0
    print(
        f"   loaded in {t_cc_load:.1f}s; n_latents={crosscoder.n_latents}, "
        f"n_hookpoints={crosscoder.n_hookpoints}"
    )
    print()

    # --- Greedy generate timing ---
    print("[3/4] Timing greedy generate() ...")
    inputs = tokenizer(
        _spaced(SMOKE_INPUT_SEQ),
        add_special_tokens=True, return_tensors="pt", padding=False,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    gk = _gen_kwargs(len(SMOKE_INPUT_SEQ))

    # warm-up
    for i in range(n_warm):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gk)
        print(f"   warmup {i+1}: {time.perf_counter() - t0:.2f}s, out_len={out.shape[1]}")

    timings = []
    decoded = None
    for i in range(n_timed):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gk)
        dt = time.perf_counter() - t0
        timings.append(dt)
        decoded = _decode_protT5_output(tokenizer, out[0])
        print(f"   timed {i+1}: {dt:.2f}s, out_len={out.shape[1]}")

    mean_gen = sum(timings) / len(timings)
    print(f"   mean greedy gen ({len(SMOKE_INPUT_SEQ)} AA): {mean_gen:.2f}s")
    print()

    # --- Copy-artifact check ---
    print("[4/4] Copy-artifact diagnostic")
    print(f"   input    ({len(SMOKE_INPUT_SEQ):3d}): {SMOKE_INPUT_SEQ}")
    print(f"   output   ({len(decoded):3d}): {decoded}")
    n_match = sum(1 for a, b in zip(SMOKE_INPUT_SEQ, decoded) if a == b)
    n_aligned = min(len(SMOKE_INPUT_SEQ), len(decoded))
    identity = n_match / n_aligned if n_aligned > 0 else 0.0
    print(
        f"   aligned identity (first {n_aligned} positions): "
        f"{n_match}/{n_aligned} = {100*identity:.1f}%"
    )
    if identity > 0.95:
        print("   --> COPY ARTIFACT confirmed.")
    elif identity > 0.5:
        print("   --> partial copy.")
    else:
        print("   --> no copy. (Important: the decoder may not be in copy mode.)")
    print()

    # --- Crosscoder encode timing (per-residue, in encoder forward hook) ---
    print("[Bonus] Crosscoder encode timing (per residue, CPU)")
    # Simulate: 56 residues x 24 hookpoints x 1024 d_model fp32
    L = len(SMOKE_INPUT_SEQ)
    fake_acts = torch.randn(L, 1, crosscoder.n_hookpoints, crosscoder.d_model)
    t0 = time.perf_counter()
    with torch.no_grad():
        z = crosscoder.get_pre_bias_BL(fake_acts)
        if crosscoder.b_enc_L is not None:
            z = z + crosscoder.b_enc_L
        z = crosscoder.activation_fn.forward(z)
        recon = crosscoder.decode_BMPD(z)
    t_cc_fwd = time.perf_counter() - t0
    print(
        f"   encode+decode for L={L} residues: {t_cc_fwd*1000:.1f} ms "
        f"({1000*t_cc_fwd/L:.2f} ms/residue)"
    )
    print()

    # --- Go/no-go verdict ---
    print("=" * 60)
    print("Verdict")
    print("=" * 60)
    go = mean_gen <= 180  # 3 min per generation for a 56-AA input is the wall.
    if go:
        print(f"GO: mean generation = {mean_gen:.1f}s for L=56. "
              "Pipeline is M1-viable for the smoke test.")
    else:
        print(f"NO-GO: mean generation = {mean_gen:.1f}s for L=56 is too slow. "
              "Switch generation to H100 before proceeding.")
    print("=" * 60)

    # Cleanup
    del model, crosscoder
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "device": str(device),
        "t_prott5_load_sec": t_prott5_load,
        "t_crosscoder_load_sec": t_cc_load,
        "mem_after_prott5_bytes": mem_after_prott5,
        "mean_greedy_gen_sec": mean_gen,
        "input_seq": SMOKE_INPUT_SEQ,
        "output_seq": decoded,
        "copy_identity": identity,
        "crosscoder_encode_ms_per_residue": 1000 * t_cc_fwd / L,
        "go": go,
    }


if __name__ == "__main__":
    import fire

    fire.Fire(m1_viability)
