from typing import Any

import torch
from einops import rearrange
from torch import nn

from crosscode.models.acausal_crosscoder import ModelHookpointAcausalCrosscoder
from crosscode.models.initialization.init_strategy import InitStrategy
from crosscode.utils import l2_norm


class DataPointInit(InitStrategy[ModelHookpointAcausalCrosscoder[Any]]):
    def __init__(self, data_points: torch.Tensor, datapoint_scale: float, dec_init_norm: float):
        """
        Initializes crosscoder weights using Data Point Initialization (DPI).
        Args:
            data_points: Tensort of shape [n_latents, n_models, n_hookpoints, d_model]
            datapoint_scale: Weight to blend the data points with random noise (1.0 = pure data, 0.0 = pure random)
            dec_init_norm: Target l2 norm to normalize the decoder weights to
        """
        super().__init__()
        self.data_points = data_points
        self.datapoint_scale = datapoint_scale
        self.dec_init_norm = dec_init_norm

    @torch.no_grad()
    def init_weights(self, cc: ModelHookpointAcausalCrosscoder[Any]) -> None:
        """
        w = datapoint_scale * (x - mu) + (1 - datapoint_scale) * r
        where x are randomly chosen real data points, mu is their mean, and r is random normal (Kaiming).
        W_enc = Concat(w)
        W_dec = W_enc^T
        """
        # data_points: [n_latents, n_models, n_hookpoints, d_model] = [L, M, P, D]
        n_latents = self.data_points.shape[0]
        assert n_latents == cc.n_latents, f"data_points must have {cc.n_latents} latents, got {n_latents}"

        mu_MPD = self.data_points.mean(dim=0, keepdim=True)
        x_centered_LMPD = self.data_points - mu_MPD

        r_LMPD = torch.empty_like(x_centered_LMPD)
        nn.init.kaiming_uniform_(r_LMPD)

        w_LMPD = self.datapoint_scale * x_centered_LMPD + (1.0 - self.datapoint_scale) * r_LMPD

        # normalize to dec_init_norm
        w_LMPD.div_(l2_norm(w_LMPD, dim=-1, keepdim=True))
        w_LMPD.mul_(self.dec_init_norm)

        cc.W_dec_LMPD.copy_(w_LMPD)
        cc.W_enc_MPDL.copy_(rearrange(cc.W_dec_LMPD.clone(), "l ... -> ... l"))

        if cc.b_enc_L is not None:
            cc.b_enc_L.zero_()
        if cc.b_dec_MPD is not None:
            cc.b_dec_MPD.zero_()
