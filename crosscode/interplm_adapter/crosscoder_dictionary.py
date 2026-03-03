from typing import Optional, List
import torch
import torch.nn as nn

from interplm.sae.dictionary import Dictionary
from crosscode.models.base_crosscoder import BaseCrosscoder


class CrosscoderDictionaryWrapper(Dictionary):
    """
    Adapter that wraps a trained ModelHookpointAcausalCrosscoder so that it can be
    used by InterPLM's analysis tools.
    """

    def __init__(self, crosscoder: BaseCrosscoder, normalize_to_sqrt_d: bool = False):
        super().__init__(normalize_to_sqrt_d)
        self.crosscoder = crosscoder
        self.dict_size = crosscoder.n_latents

    def encode(self, x: torch.Tensor, normalize_features: bool = False) -> torch.Tensor:
        """
        Encode inputs to latents.
        x expected shape: [Batch, M, P, D]
        returns: [Batch, n_latents]
        """
        with torch.no_grad():
            pre_acts = self.crosscoder.get_pre_bias_BL(x)
            if self.crosscoder.b_enc_L is not None:
                pre_acts += self.crosscoder.b_enc_L

            latents = self.crosscoder.activation_fn.forward(pre_acts)
            return latents

    def encode_feat_subset(
        self, x: torch.Tensor, feat_list: List[int], normalize_features: bool = False
    ) -> torch.Tensor:
        """
        Encode only a subset of features.
        """
        with torch.no_grad():
            W_enc_subset = self.crosscoder._W_enc_XiDiL[
                ..., feat_list
            ]  # [M, P, D, subset]
            b_enc_subset = (
                self.crosscoder.b_enc_L[feat_list]
                if self.crosscoder.b_enc_L is not None
                else None
            )

            # einsum B ..., ... l -> B l
            pre_acts = torch.einsum("b... , ...l -> bl", x, W_enc_subset)

            if b_enc_subset is not None:
                pre_acts += b_enc_subset

            all_latents = self.encode(x, normalize_features=normalize_features)
            return all_latents[:, feat_list]

    def forward(self, x, output_features=False, ghost_mask=None, unnormalize=False):
        """Not heavily used during analysis, but included for completeness."""
        with torch.no_grad():
            res = self.crosscoder._forward_train(x)
            if output_features:
                return res.output_BXoDo, res.latents_BL
            return res.output_BXoDo

    @classmethod
    def from_pretrained(cls, path, device=None):
        raise NotImplementedError(
            "Use load_sae from crosscode.interplm_adapter.load_sae to load crosscoders."
        )
