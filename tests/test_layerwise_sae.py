"""The shared loop of topk_layerwise_sae trains each SAE as BaseTrainer.train does.

Two SAEs with the same init train on the same batches, once with BaseTrainer.train (one hookpoint
each) and once with train_layerwise (two hookpoints from one dataloader). The weights must be
identical. A dead-latent threshold of 50 tokens makes AuxK active. CPU, synthetic data.
"""

import torch

from crosscode.data.activations_dataloader import ModelHookpointActivationsBatch
from crosscode.models import AnthropicTransposeInit, ModelHookpointAcausalCrosscoder
from crosscode.models.activations.topk import BatchTopkActivation
from crosscode.trainers.topk_crosscoder.config import TopKTrainConfig
from crosscode.trainers.topk_crosscoder.trainer import TopKStyleAcausalCrosscoderTrainer
from crosscode.trainers.topk_layerwise_sae.run import OneHookpointView, train_layerwise

D_MODEL, N_LATENTS, B, N_STEPS = 32, 64, 16, 60
HOOKPOINTS = ["encoder.blocks.7.hook_resid_post", "encoder.blocks.16.hook_resid_post"]


class _StubRun:
    id = "stub"
    summary: dict = {}

    def log(self, *args, **kwargs): ...

    def finish(self): ...


class _StubDataloader:
    def __init__(self, batches_BMPD: list[torch.Tensor], scaling_MP: torch.Tensor, hookpoints: list[str]):
        self._batches, self._scaling_MP = batches_BMPD, scaling_MP
        self.n_models, self.hookpoints, self.n_hookpoints = 1, hookpoints, len(hookpoints)

    def get_scaling_factors(self) -> torch.Tensor:
        return self._scaling_MP

    def get_activations_iterator(self):
        return (ModelHookpointActivationsBatch(batch) for batch in self._batches)


def _sae() -> ModelHookpointAcausalCrosscoder:
    return ModelHookpointAcausalCrosscoder(
        n_models=1,
        n_hookpoints=1,
        d_model=D_MODEL,
        n_latents=N_LATENTS,
        activation_fn=BatchTopkActivation(k_per_example=4),
        init_strategy=AnthropicTransposeInit(dec_init_norm=0.1),
    )


def test_train_layerwise_matches_base_trainer(tmp_path):
    torch.manual_seed(0)
    batches_BMPD = [torch.randn(B, 1, len(HOOKPOINTS), D_MODEL) for _ in range(N_STEPS)]
    scaling_MP = torch.tensor([[0.5, 2.0]])
    cfg = TopKTrainConfig(
        batch_size=B,
        num_steps=N_STEPS,
        log_every_n_steps=10,
        save_every_n_steps=25,
        topk_style="batch_topk",
        dead_latents_threshold_n_examples=50,
        k_aux=16,
    )
    shared = _StubDataloader(batches_BMPD, scaling_MP, HOOKPOINTS)
    cpu = torch.device("cpu")
    layerwise = []
    for p, hookpoint in enumerate(HOOKPOINTS):
        init_state = _sae().state_dict()
        reference = _sae()
        reference.load_state_dict(init_state)
        one_layer = _StubDataloader([b[:, :, p : p + 1] for b in batches_BMPD], scaling_MP[:, p : p + 1], [hookpoint])
        TopKStyleAcausalCrosscoderTrainer(cfg, one_layer, reference, _StubRun(), cpu, tmp_path / f"ref{p}").train()

        sae = _sae()
        sae.load_state_dict(init_state)
        trainer = TopKStyleAcausalCrosscoderTrainer(
            cfg, OneHookpointView(shared, p), sae, _StubRun(), cpu, tmp_path / f"new{p}"
        )
        layerwise.append((reference, sae, trainer))

    train_layerwise(shared, [trainer for _, _, trainer in layerwise])

    for reference, sae, trainer in layerwise:
        assert trainer.step == N_STEPS
        for name, value in reference.state_dict().items():
            assert torch.equal(value, sae.state_dict()[name]), name
