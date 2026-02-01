import re

import torch
from Bio import SeqIO
from transformers import T5EncoderModel, T5Tokenizer

from crosscode.data.activation_harvester import ActivationsHarvester


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def test_prott5_verif():
    device = get_device()
    model_name = "Rostlab/prot_t5_xl_uniref50"
    print(f"Loading {model_name} on {device}...")

    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to(device)
    model.eval()

    fasta_path = "dataset/uniref50_3M_length_512.fasta"
    print(f"Loading sequences from {fasta_path}...")
    records = list(SeqIO.parse(fasta_path, "fasta"))
    sequences = [str(record.seq) for record in records[:2]]  # Just 2 for speed

    # Pre-processing
    processed_seqs = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in sequences]
    ids = tokenizer.batch_encode_plus(processed_seqs, add_special_tokens=True, padding="longest", return_tensors="pt")
    input_ids = ids["input_ids"].to(device)
    attention_mask = ids["attention_mask"].to(device)

    # 1. Reference Method using Manual Hook (The "True" Pre-Norm state)
    ref_cache = {}

    def ref_hook(module, input, output):
        # T5Block output is (hidden_states, [attention_outputs])
        ref_cache["act"] = output[0].detach().cpu() if isinstance(output, tuple) else output.detach().cpu()

    # Hook the 24th block (index 23)
    target_block = model.encoder.block[23]
    handle = target_block.register_forward_hook(ref_hook)

    print("Running reference inference...")
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    handle.remove()

    recommended_acts = ref_cache["act"]
    print(f"Reference acts shape: {recommended_acts.shape}")
    print(f"Reference acts mean magnitude: {recommended_acts.abs().mean().item():.4f}")

    # 2. Harvester Method
    print("Running harvester inference...")
    hookpoints = ["encoder.blocks.24.hook_resid_post"]
    harvester = ActivationsHarvester(llms=[model], hookpoints=hookpoints)

    # Note: we call harvester.get_activations_HSMPD which does NOT include dataloader scaling
    harvester_acts_HSMPD = harvester.get_activations_HSMPD(input_ids)
    harvester_acts = harvester_acts_HSMPD[:, :, 0, 0, :].detach().cpu()
    print(f"Harvester acts shape: {harvester_acts.shape}")
    print(f"Harvester acts mean magnitude: {harvester_acts.abs().mean().item():.4f}")

    # 3. Compare
    diff = torch.abs(recommended_acts - harvester_acts).max().item()
    print(f"Max difference: {diff}")

    # Check against post-norm state just for forensics
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    post_norm_acts = out.last_hidden_state.detach().cpu()
    post_norm_diff = torch.abs(post_norm_acts - harvester_acts).max().item()
    print(f"Difference vs Post-Norm state: {post_norm_diff}")

    if diff < 1e-4:
        print("SUCCESS: Harvester matches pre-norm residual stream exactly!")
    else:
        print("FAILURE: There is still a mismatch.")
        assert diff < 1e-4


if __name__ == "__main__":
    test_prott5_verif()
