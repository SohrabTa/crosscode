"""Roundtrip test for chunked-resume training state.

Validates that BaseTrainer.save_train_state -> load_train_state faithfully
restores model weights, optimizer moments, global step/epoch/token counters, the
firing tracker, and the frozen norm scaling factors. This is the load-bearing
correctness for chunked full-UniRef50 training (each chunk resumes the previous).

Runs on CPU with tiny dims; no ProtT5/GPU needed.
"""
from pathlib import Path

import torch

from crosscode.models import AnthropicTransposeInit, ModelHookpointAcausalCrosscoder
from crosscode.models.activations.topk import BatchTopkActivation
from crosscode.trainers.topk_crosscoder.config import TopKTrainConfig
from crosscode.trainers.topk_crosscoder.trainer import TopKStyleAcausalCrosscoderTrainer

D_MODEL, N_LATENTS, K = 8, 16, 2
N_MODELS, N_HOOKPOINTS, B = 1, 2, 4


class _StubDataloader:
    """Minimal dataloader satisfying the trainer's asserts + get_scaling_factors."""

    def __init__(self, scaling_MP: torch.Tensor):
        self.n_models = N_MODELS
        self.n_hookpoints = N_HOOKPOINTS
        self.hookpoints = [f"hp{i}" for i in range(N_HOOKPOINTS)]
        self._scaling_MP = scaling_MP

    def get_scaling_factors(self) -> torch.Tensor:
        return self._scaling_MP


class _StubWandb:
    def __init__(self, run_id="run_abc123"):
        self.summary = {}
        self.id = run_id

    def log(self, *a, **k): ...
    def finish(self, *a, **k): ...


def _build_trainer(save_dir: Path, scaling_MP: torch.Tensor):
    model = ModelHookpointAcausalCrosscoder(
        n_models=N_MODELS,
        n_hookpoints=N_HOOKPOINTS,
        d_model=D_MODEL,
        n_latents=N_LATENTS,
        init_strategy=AnthropicTransposeInit(dec_init_norm=0.1),
        activation_fn=BatchTopkActivation(k_per_example=K),
        use_encoder_bias=True,
        use_decoder_bias=True,
    )
    cfg = TopKTrainConfig(
        batch_size=B,
        num_steps=1000,
        log_every_n_steps=10,
        topk_style="batch_topk",
        k_aux=4,
        lambda_aux=0.03125,
        dead_latents_threshold_n_examples=100,
    )
    return TopKStyleAcausalCrosscoderTrainer(
        cfg=cfg,
        activations_dataloader=_StubDataloader(scaling_MP),
        model=model,
        wandb_run=_StubWandb(),
        device=torch.device("cpu"),
        save_dir=save_dir,
    )


def test_resume_roundtrip(tmp_path: Path):
    torch.manual_seed(0)
    scaling_MP = torch.rand(N_MODELS, N_HOOKPOINTS) + 0.5
    t1 = _build_trainer(tmp_path / "run1", scaling_MP)

    # Take a couple of real optimizer steps so weights + Adam moments are non-trivial.
    for _ in range(3):
        batch = torch.randn(B, N_MODELS, N_HOOKPOINTS, D_MODEL)
        res = t1.model.forward_train(batch)
        t1.firing_tracker.add_batch(res.latents_BL)
        loss, _ = t1._calculate_loss_and_log(batch, res, log=False)
        t1.optimizer.zero_grad()
        loss.backward()
        t1.optimizer.step()
        t1.step += 1
    t1.unique_tokens_trained = 12345
    t1.epoch = 0

    ckpt = tmp_path / "ckpt"
    t1.save_train_state(ckpt)
    assert (ckpt / t1.TRAIN_STATE_FNAME).exists()

    # Fresh trainer with DIFFERENT init + scaling, then resume.
    t2 = _build_trainer(tmp_path / "run2", torch.ones(N_MODELS, N_HOOKPOINTS))
    # sanity: before load, weights differ
    p1 = dict(t1.model.named_parameters())
    p2_before = dict(t2.model.named_parameters())
    assert any(not torch.allclose(p1[k], p2_before[k]) for k in p1)

    t2.load_train_state(ckpt)

    # 1. model weights match
    p2 = dict(t2.model.named_parameters())
    for k in p1:
        assert torch.allclose(p1[k], p2[k]), f"param {k} mismatch after resume"
    # 2. counters match
    assert t2.step == t1.step == 3
    assert t2.epoch == t1.epoch
    assert t2.unique_tokens_trained == 12345
    # 3. firing tracker matches
    assert torch.equal(t2.firing_tracker.tokens_since_fired_L, t1.firing_tracker.tokens_since_fired_L)
    # 4. optimizer Adam moments (exp_avg / exp_avg_sq) match
    s1 = t1.optimizer.state_dict()["state"]
    s2 = t2.optimizer.state_dict()["state"]
    assert set(s1) == set(s2) and len(s1) > 0
    for pid in s1:
        for mkey in ("exp_avg", "exp_avg_sq"):
            if mkey in s1[pid]:
                assert torch.allclose(s1[pid][mkey], s2[pid][mkey]), f"optimizer {mkey} mismatch"
    # 5. frozen norm scaling factors roundtrip via the classmethod used by the dataloader
    norm = TopKStyleAcausalCrosscoderTrainer.load_norm_scaling_factors(ckpt)
    assert torch.allclose(norm, scaling_MP)
    # 6. wandb run id roundtrips so the next chunk resumes the same run
    assert TopKStyleAcausalCrosscoderTrainer.load_wandb_run_id(ckpt) == "run_abc123"

    print("RESUME ROUNDTRIP OK: weights, step, firing tracker, optimizer moments, "
          "norm factors, wandb run id all match")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_resume_roundtrip(Path(d))
