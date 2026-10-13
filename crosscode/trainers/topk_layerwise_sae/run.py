"""Train one BatchTopK SAE per hookpoint from one shared ProtT5 harvest (roadmap PP-06a).

Each SAE is a ModelHookpointAcausalCrosscoder with one hookpoint. TopKStyleAcausalCrosscoderTrainer
trains it with the loss, AuxK, normalization and LR schedule of the crosscoder. The SAEs share the
ProtT5 forward pass and the shuffled batches, and nothing else. Each SAE has its own parameters,
optimizer, firing tracker and save dir (<save_dir>/sae<p>_<hookpoint>/).

The config is a TopKAcausalCrosscoderExperimentConfig. `hookpoints` lists the SAE layers, and
`crosscoder` gives the settings of each SAE. A hookpoint that occurs two times gives two SAEs with
different initializations (a seed-variance pair). There is no chunked resume. The run ends at
num_steps or when the data runs out, whichever comes first.

    uv run crosscode/trainers/topk_layerwise_sae/run.py training_config_layerwise_sae.yaml
"""

import time

import fire
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm  # type: ignore
from transformers import T5EncoderModel

from crosscode.data.activations_dataloader import (
    ActivationsDataloader,
    ModelHookpointActivationsBatch,
    ModelHookpointActivationsDataloader,
    build_model_hookpoint_dataloader,
)
from crosscode.llms import build_llms
from crosscode.log import logger
from crosscode.models import AnthropicTransposeInit, ModelHookpointAcausalCrosscoder
from crosscode.models.activations.topk import BatchTopkActivation
from crosscode.trainers.base_trainer import run_exp
from crosscode.trainers.topk_crosscoder.config import TopKAcausalCrosscoderExperimentConfig
from crosscode.trainers.topk_crosscoder.trainer import TopKStyleAcausalCrosscoderTrainer
from crosscode.trainers.utils import build_wandb_run
from crosscode.utils import get_device


class OneHookpointView(ActivationsDataloader[ModelHookpointActivationsBatch]):
    """Hookpoint `p` of the shared dataloader, as the trainer of one SAE sees it.

    The trainer reads the shape, the hookpoint name and the norm factor from this view. The batches
    come from train_layerwise, which slices them from the shared dataloader.
    """

    def __init__(self, shared: ModelHookpointActivationsDataloader, p: int):
        self._shared, self._p = shared, p
        self.n_models = shared.n_models
        self.hookpoints = [shared.hookpoints[p]]
        self.n_hookpoints = 1

    def get_scaling_factors(self) -> torch.Tensor:
        return self._shared.get_scaling_factors()[:, self._p : self._p + 1]

    def get_activations_iterator(self):
        raise NotImplementedError("train_layerwise gives the batches to the trainers")


def build_layerwise(
    cfg: TopKAcausalCrosscoderExperimentConfig,
) -> tuple[ModelHookpointActivationsDataloader, list[TopKStyleAcausalCrosscoderTrainer]]:
    if cfg.train.topk_style != "batch_topk" or cfg.train.gradient_accumulation_steps_per_batch != 1:
        raise ValueError("train_layerwise supports batch_topk without gradient accumulation only")
    if cfg.train.resume_from is not None:
        raise ValueError("train_layerwise has no resume. Size num_steps to one job")
    torch.manual_seed(cfg.seed)
    logger.info(f"torch.manual_seed({cfg.seed}) for the SAE initialization")
    logger.info(f"torch {torch.__version__}, float32_matmul_precision={torch.get_float32_matmul_precision()}")

    device = get_device()
    harvester_cfg = cfg.data.activations_harvester
    llms = build_llms(harvester_cfg.llms, cfg.cache_dir, device, inferenced_type=harvester_cfg.inference_dtype)
    d_model = llms[0].config.d_model if isinstance(llms[0], T5EncoderModel) else llms[0].cfg.d_model
    dataloader = build_model_hookpoint_dataloader(
        cfg=cfg.data, llms=llms, hookpoints=cfg.hookpoints, batch_size=cfg.train.batch_size, cache_dir=cfg.cache_dir
    )
    norm_factors = list(zip(cfg.hookpoints, dataloader.get_scaling_factors()[0].tolist(), strict=True))
    logger.info(f"norm scaling factors {norm_factors}")

    wandb_run = build_wandb_run(cfg)
    trainers = []
    for p, hookpoint in enumerate(cfg.hookpoints):
        sae = ModelHookpointAcausalCrosscoder(
            n_models=len(llms),
            n_hookpoints=1,
            d_model=d_model,
            n_latents=cfg.crosscoder.n_latents,
            init_strategy=AnthropicTransposeInit(dec_init_norm=cfg.crosscoder.dec_init_norm),
            activation_fn=BatchTopkActivation(k_per_example=cfg.crosscoder.k),
            use_encoder_bias=cfg.crosscoder.use_encoder_bias,
            use_decoder_bias=cfg.crosscoder.use_decoder_bias,
        ).to(device)
        view = OneHookpointView(dataloader, p)
        save_dir = cfg.save_dir / f"sae{p}_{hookpoint}"
        trainers.append(TopKStyleAcausalCrosscoderTrainer(cfg.train, view, sae, wandb_run, device, save_dir))
    return dataloader, trainers


def train_layerwise(
    dataloader: ModelHookpointActivationsDataloader, trainers: list[TopKStyleAcausalCrosscoderTrainer]
) -> None:
    """Give each batch of the shared dataloader to all SAEs, one hookpoint slice to each."""
    train_cfg, wandb_run = trainers[0].cfg, trainers[0].wandb_run
    batches = dataloader.get_activations_iterator()
    start = time.perf_counter()
    try:
        for step in tqdm(range(train_cfg.num_steps), desc="Train Steps", smoothing=0.15):
            try:
                batch_BMPD = next(batches).activations_BMPD
            except StopIteration:
                logger.info("Dataloader ran out of data. Finishing training safely.")
                break
            log = step % train_cfg.log_every_n_steps == 0
            log_dict: dict[str, float] = {}
            for p, trainer in enumerate(trainers):
                layer_logs = _train_step(trainer, batch_BMPD[:, :, p : p + 1], log)
                log_dict.update({f"{trainer.save_dir.name}/{key}": value for key, value in layer_logs.items()})
            if log:
                wandb_run.log({**log_dict, "perf/elapsed_s": time.perf_counter() - start}, step=step)
            if step + 1 in (100, 1_000, 10_000) or (step + 1) % 100_000 == 0:
                _log_progress(step + 1, time.perf_counter() - start)
    except BaseException:
        for trainer in trainers:
            trainer._save_final_or_crash_model(is_crash=True)
        wandb_run.finish()
        raise
    for trainer in trainers:
        trainer._save_final_or_crash_model(is_crash=False)
    wandb_run.finish()


def _log_progress(n_steps: int, elapsed_s: float) -> None:
    """Log the sustained rate (harvest stalls included) and the peak GPU memory, so that a slow
    node or a memory problem shows in the job log after the first minutes."""
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    rate = n_steps / elapsed_s
    logger.info(
        f"progress: {n_steps:,} steps in {elapsed_s:,.0f} s = {rate:.2f} steps/s sustained, "
        f"peak GPU memory {peak_gb:.1f} GB, {1e6 / rate / 3600:.1f} h per million steps"
    )


def _train_step(trainer: TopKStyleAcausalCrosscoderTrainer, batch_BMPD: torch.Tensor, log: bool) -> dict:
    """One optimizer step of one SAE, in the order of BaseTrainer.train (base_trainer.py:157-202)."""
    trainer._lr_step()
    trainer.optimizer.zero_grad()
    loss, log_dict, n_tokens = trainer.run_batch(ModelHookpointActivationsBatch(batch_BMPD), log)
    trainer.unique_tokens_trained += n_tokens
    loss.backward()
    logs = {**log_dict, **trainer._step_logs()} if log_dict is not None else {}
    trainer._maybe_save_model()
    clip_grad_norm_(trainer.model.parameters(), 1.0)
    trainer.optimizer.step()
    trainer.step += 1
    return logs


class LayerwiseSAETrainer:
    """The object that run_exp builds and then trains (base_trainer.py:279-281)."""

    def __init__(self, cfg: TopKAcausalCrosscoderExperimentConfig):
        self.dataloader, self.trainers = build_layerwise(cfg)

    def train(self) -> None:
        train_layerwise(self.dataloader, self.trainers)


if __name__ == "__main__":
    fire.Fire(run_exp(LayerwiseSAETrainer, TopKAcausalCrosscoderExperimentConfig))
