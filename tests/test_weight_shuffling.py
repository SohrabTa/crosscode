import torch
from transformers import AutoModelForCausalLM
from crosscode.llms import shuffle_model_weights
from crosscode.trainers.config_common import LLMConfig

def test_shuffle_helper_function():
    print("Testing shuffle_model_weights on a simple linear layer...")
    # Create a simple mock model
    model = torch.nn.Linear(100, 100)
    
    # Save original weights
    orig_weight = model.weight.clone()
    orig_bias = model.bias.clone()
    
    # Shuffle
    shuffle_model_weights(model)
    
    # Check shape
    assert model.weight.shape == orig_weight.shape, "Weight shape changed!"
    assert model.bias.shape == orig_bias.shape, "Bias shape changed!"
    
    # Check values changed
    assert not torch.allclose(model.weight, orig_weight), "Weights were not shuffled!"
    assert not torch.allclose(model.bias, orig_bias), "Biases were not shuffled!"
    
    # Check values are exactly the same set of values (just permuted)
    # We can do this by sorting the flattened tensors and comparing
    sorted_orig_weight, _ = torch.sort(orig_weight.view(-1))
    sorted_shuffled_weight, _ = torch.sort(model.weight.view(-1))
    assert torch.allclose(sorted_orig_weight, sorted_shuffled_weight), "Weight values distribution changed!"
    
    sorted_orig_bias, _ = torch.sort(orig_bias.view(-1))
    sorted_shuffled_bias, _ = torch.sort(model.bias.view(-1))
    assert torch.allclose(sorted_orig_bias, sorted_shuffled_bias), "Bias values distribution changed!"
    
    print("✅ shuffle_model_weights helper function works perfectly.")

from transformers import T5EncoderModel

def test_real_model_shuffling():
    print("\nTesting full model shuffling (using ProtT5)...")
    model_name = "Rostlab/prot_t5_xl_uniref50"
    
    print(f"Loading original {model_name}...")
    orig_model = T5EncoderModel.from_pretrained(model_name)
    
    print(f"Loading another instance and shuffling with seed 42...")
    shuffled_model = T5EncoderModel.from_pretrained(model_name)
    shuffle_model_weights(shuffled_model, seed=42)
    
    print(f"Loading a third instance and shuffling with seed 42 to verify reproducibility...")
    shuffled_model_reproducible = T5EncoderModel.from_pretrained(model_name)
    shuffle_model_weights(shuffled_model_reproducible, seed=42)
    
    # Compare a specific layer's weights
    orig_w = orig_model.encoder.block[0].layer[0].SelfAttention.q.weight.data
    shuf_w = shuffled_model.encoder.block[0].layer[0].SelfAttention.q.weight.data
    shuf_w_repro = shuffled_model_reproducible.encoder.block[0].layer[0].SelfAttention.q.weight.data
    
    assert orig_w.shape == shuf_w.shape, "Shape mismatch in real model!"
    assert not torch.allclose(orig_w, shuf_w), "Real model weights were not shuffled!"
    assert torch.allclose(shuf_w, shuf_w_repro), "Randomization with same seed is not reproducible!"
    
    sorted_orig, _ = torch.sort(orig_w.view(-1))
    sorted_shuf, _ = torch.sort(shuf_w.view(-1))
    assert torch.allclose(sorted_orig, sorted_shuf), "Real model weight distribution changed!"
    
    print("✅ Real model shuffling works perfectly, preserves exact weight values, and is reproducible.")

if __name__ == "__main__":
    test_shuffle_helper_function()
    test_real_model_shuffling()
    print("\n🎉 All tests passed! The shuffling mechanism behaves exactly as required by InterPLM.")
