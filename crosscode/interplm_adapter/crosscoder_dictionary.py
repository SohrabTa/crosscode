import torch
from interplm.sae.dictionary import Dictionary

from crosscode.models.base_crosscoder import BaseCrosscoder


class CrosscoderDictionaryWrapper(Dictionary):
    """
    Adapter that wraps a trained ModelHookpointAcausalCrosscoder so that it can be
    used by InterPLM's analysis tools.

    Per-feature activation normalization
    ------------------------------------
    InterPLM's analysis tooling (collect_feature_activations, dashboard,
    LLM annotation) expects feature activations on a [0, 1] scale where 1.0
    is the per-feature maximum observed by ``normalize.py``. Those scaling
    factors are saved in ``activation_rescale_factor`` inside
    ``ae_normalized.pt``.

    This wrapper:

    * Registers ``activation_rescale_factor`` as a buffer in ``__init__`` so
      that ``load_state_dict`` (called by ``load_sae``) populates it instead
      of silently dropping the tensor as ``strict=False``-unexpected.
    * Applies ``features / activation_rescale_factor`` in ``encode`` and
      ``encode_feat_subset`` whenever the caller passes
      ``normalize_features=True``.

    Without these two changes any analysis that requests normalized features
    (i.e. ``collect_feature_activations.py``) would receive raw, un-rescaled
    activations and bin thresholds intended for the ``[0, 1]`` range would be
    applied to arbitrary absolute values.
    """

    def __init__(self, crosscoder: BaseCrosscoder, normalize_to_sqrt_d: bool = False):
        super().__init__(normalize_to_sqrt_d)
        self.crosscoder = crosscoder
        self.dict_size = crosscoder.n_latents
        self.activation_dim = crosscoder.d_model
        # Buffer defaults to ones, so calling ``encode(..., normalize_features=True)``
        # on an unnormalized checkpoint is a no-op rather than a crash. Real
        # rescale factors are loaded from ``ae_normalized.pt`` by
        # ``Dictionary._load_from_state_dict``.
        self.register_buffer(
            "activation_rescale_factor", torch.ones(crosscoder.n_latents)
        )

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to reconstructions.
        f expected shape: [Batch, n_latents]
        returns: [Batch, M, P, D]
        """
        with torch.no_grad():
            return self.crosscoder.decode_BMPD(f)

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
            if normalize_features:
                # Clamp denominator so that dead features (rescale == 0,
                # latent always 0) yield 0 instead of NaN.
                latents = latents / self.activation_rescale_factor.clamp(min=1e-12)
            return latents

    def encode_feat_subset(
        self, x: torch.Tensor, feat_list: list[int], normalize_features: bool = False
    ) -> torch.Tensor:
        """
        Encode only a subset of features.

        We apply normalization on the subset itself (rather than re-encoding
        with normalize_features=True and slicing) to avoid an unnecessary
        full-dictionary division.
        """
        with torch.no_grad():
            all_latents = self.encode(x, normalize_features=False)
            features = all_latents[:, feat_list]
            if normalize_features:
                features = features / self.activation_rescale_factor[feat_list].clamp(
                    min=1e-12
                )
            return features

    def forward(self, x, output_features=False, ghost_mask=None, unnormalize=False):
        """Not heavily used during analysis, but included for completeness."""
        with torch.no_grad():
            res = self.crosscoder._forward_train(x)
            if output_features:
                return res.output_BXoDo, res.latents_BL
            return res.output_BXoDo

    @classmethod
    def from_pretrained(cls, path, device=None):
        raise NotImplementedError("Use load_sae from crosscode.interplm_adapter.load_sae to load crosscoders.")
