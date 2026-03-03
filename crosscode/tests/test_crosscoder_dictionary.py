import torch
import unittest
from crosscode.models.acausal_crosscoder import ModelHookpointAcausalCrosscoder
from crosscode.interplm_adapter.crosscoder_dictionary import CrosscoderDictionaryWrapper
from crosscode.models.activations import BatchTopKActivation


class TestCrosscoderDictionaryWrapper(unittest.TestCase):
    def setUp(self):
        self.b = 4
        self.n_models = 1
        self.n_hookpoints = 24
        self.d_model = 16
        self.n_latents = 32

        activation_fn = BatchTopKActivation(k=2)

        self.crosscoder = ModelHookpointAcausalCrosscoder(
            n_models=self.n_models,
            n_hookpoints=self.n_hookpoints,
            d_model=self.d_model,
            n_latents=self.n_latents,
            activation_fn=activation_fn,
            use_encoder_bias=True,
            use_decoder_bias=False,
        )
        self.wrapper = CrosscoderDictionaryWrapper(self.crosscoder)

    def test_encode_shape(self):
        dummy_input = torch.randn(
            self.b, self.n_models, self.n_hookpoints, self.d_model
        )
        latents = self.wrapper.encode(dummy_input)

        self.assertEqual(latents.shape, (self.b, self.n_latents))

    def test_encode_feat_subset(self):
        dummy_input = torch.randn(
            self.b, self.n_models, self.n_hookpoints, self.d_model
        )
        feat_list = [0, 5, 10]
        subset_latents = self.wrapper.encode_feat_subset(dummy_input, feat_list)

        self.assertEqual(subset_latents.shape, (self.b, len(feat_list)))

        full_latents = self.wrapper.encode(dummy_input)
        self.assertTrue(
            torch.allclose(subset_latents, full_latents[:, feat_list], atol=1e-5)
        )


if __name__ == "__main__":
    unittest.main()
