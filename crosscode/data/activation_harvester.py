from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import torch
from einops import pack
from transformer_lens import HookedTransformer  # type: ignore
from transformers import T5EncoderModel  # type: ignore

from crosscode.data.activation_cache import ActivationsCache
from crosscode.log import logger

# shapes:
# H: (harvesting) batch size
# S: sequence length
# P: hookpoints
# D: model d_model

CacheMode = Literal["no_cache", "cache"]  # , "cache_with_mmap"] # TODO validate mmap works, not confient atm


class HarvestingStrategy(ABC):
    """Strategy for architecture-specific harvesting logic."""

    @abstractmethod
    def get_device(self, llm: torch.nn.Module) -> torch.device: ...

    @abstractmethod
    def run_with_cache(
        self,
        llm: torch.nn.Module,
        sequence_HS: torch.Tensor,
        hookpoints: list[str],
        device: torch.device,
        layer_to_stop_at: int | None,
    ) -> dict[str, torch.Tensor]: ...

    @abstractmethod
    def get_layer_index(self, hookpoint: str) -> int: ...


class TransformerStrategy(HarvestingStrategy):
    """Strategy for HookedTransformer (Decoder-only models)."""

    def get_device(self, llm: HookedTransformer) -> torch.device:
        return llm.W_E.device

    def run_with_cache(
        self,
        llm: HookedTransformer,
        sequence_HS: torch.Tensor,
        hookpoints: list[str],
        device: torch.device,
        layer_to_stop_at: int | None,
    ) -> dict[str, torch.Tensor]:
        _, cache = llm.run_with_cache(
            sequence_HS.to(device),
            names_filter=lambda name: name in hookpoints,
            stop_at_layer=layer_to_stop_at,
        )
        return cache

    def get_layer_index(self, hookpoint: str) -> int:
        return _get_layer(hookpoint)


class ProtT5Strategy(HarvestingStrategy):
    """Strategy for T5 models (e.g., ProtT5 xl)."""

    def get_device(self, llm: torch.nn.Module) -> torch.device:
        if hasattr(llm, "device"):
            return llm.device
        return next(llm.parameters()).device

    def run_with_cache(
        self,
        llm: torch.nn.Module,
        sequence_HS: torch.Tensor,
        hookpoints: list[str],
        device: torch.device,
        layer_to_stop_at: int | None,
    ) -> dict[str, torch.Tensor]:
        # If it's an EncoderDecoder model, we harvest from the encoder
        model_to_run = llm.encoder if hasattr(llm, "encoder") else llm

        attention_mask = (sequence_HS != 0).long().to(device)  # Assuming 0 is pad
        with torch.no_grad():
            output = model_to_run(
                input_ids=sequence_HS.to(device),
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        cache = {}
        for hookpoint in hookpoints:
            layer_idx = self.get_layer_index(hookpoint)
            cache[hookpoint] = output.hidden_states[layer_idx]

        return cache

    def get_layer_index(self, hookpoint: str) -> int:
        return _get_layer(hookpoint)


class ActivationsHarvester:
    def __init__(
        self,
        llms: list[HookedTransformer | T5EncoderModel],
        hookpoints: list[str],
        activations_cache_dir: Path | None = None,
        cache_mode: CacheMode = "no_cache",
    ):
        def get_d_model(llm):
            if hasattr(llm, "cfg"):
                return llm.cfg.d_model
            return llm.config.d_model

        if len({get_d_model(llm) for llm in llms}) != 1:
            raise ValueError("All models must have the same d_model")
        self._llms = llms

        # Select strategy based on the first model
        first_llm = llms[0]
        if isinstance(first_llm, T5EncoderModel) or "T5" in first_llm.__class__.__name__:
            self._strategy = ProtT5Strategy()
        else:
            self._strategy = TransformerStrategy()

        self._device = self._strategy.get_device(first_llm)
        self._hookpoints = hookpoints

        # Set up the activations cache
        self._activation_cache = None
        if cache_mode != "no_cache":
            if not activations_cache_dir:
                raise ValueError("cache_mode is enabled but no cache_dir provided; caching will be disabled")
            self._activation_cache = ActivationsCache(
                cache_dir=activations_cache_dir  # , use_mmap=cache_mode == "cache_with_mmap" # TODO validate mmap works, not confient atm
            )

        self.num_models = len(llms)
        self.num_hookpoints = len(hookpoints)
        self._layer_to_stop_at = self._get_layer_to_stop_at()

    def _get_layer_to_stop_at(self) -> int:
        last_needed_layer = max(self._strategy.get_layer_index(name) for name in self._hookpoints)
        layer_to_stop_at = last_needed_layer + 1
        logger.info(f"computed last needed layer: {last_needed_layer}, stopping at {layer_to_stop_at}")
        return layer_to_stop_at

    def _get_acts_HSPD(self, llm: HookedTransformer | T5EncoderModel, sequence_HS: torch.Tensor) -> torch.Tensor:
        if self._activation_cache is not None:
            cache_key = self._activation_cache.get_cache_key(llm, sequence_HS, self._hookpoints)
            activations_HSPD = self._activation_cache.load_activations(cache_key, self._device)
            if activations_HSPD is not None:
                return activations_HSPD

        with torch.inference_mode():
            cache = self._strategy.run_with_cache(
                llm, sequence_HS, self._hookpoints, self._device, self._layer_to_stop_at
            )

        acts_HSD = []
        for hookpoint in self._hookpoints:
            acts_HSD.append(cache[hookpoint])

        acts_HSPD, _ = pack(acts_HSD, "h s * d")

        if self._activation_cache is not None:
            cache_key = self._activation_cache.get_cache_key(llm, sequence_HS, self._hookpoints)
            self._activation_cache.save_activations(cache_key, acts_HSPD)

        return acts_HSPD

    def get_activations_HSMPD(
        self,
        sequence_HS: torch.Tensor,
    ) -> torch.Tensor:
        # MPD = (len(self._llms), len(self._hookpoints), self._llms[0].cfg.d_model)
        # activations_HSMPD = torch.empty(*sequence_HS.shape, *MPD, device=self._device)
        activations_HSPD = []  # each element is (harvest_batch_size, sequence_length, n_hookpoints)
        for model in self._llms:
            activations_HSPD.append(self._get_acts_HSPD(model, sequence_HS))

        activations_HSMPD, _ = pack(activations_HSPD, "h s * p d")
        return activations_HSMPD


def _get_layer(hookpoint: str) -> int:
    """Extracts the layer index from a hookpoint name.

    Supports formats like:
    - blocks.1.hook_resid_post
    - encoder.blocks.1.hook_resid_post
    - decoder.blocks.1.hook_resid_post
    """
    if "blocks" not in hookpoint:
        raise NotImplementedError(
            f'Hookpoint "{hookpoint}" is not a blocks hookpoint, cannot determine layer, (but feel free to add this functionality!)'
        )
    parts = hookpoint.split(".")
    try:
        blocks_idx = parts.index("blocks")
        return int(parts[blocks_idx + 1])
    except (ValueError, IndexError):
        raise ValueError(f"Could not determine layer index from hookpoint: {hookpoint}") from None
