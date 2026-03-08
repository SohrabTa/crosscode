import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

from crosscode.data.activation_harvester import ActivationsHarvester
from crosscode.utils import get_device

MODEL_DTYPE = torch.float32


def embed_annotations(
    input_dir: Path,
    output_dir: Path,
    model_name: str = "Rostlab/prot_t5_xl_uniref50",
    batch_size: int = 8,
    sequence_column: str = "sequence",
    max_sequences: int | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)

    print(f"Loading model with dtype: {MODEL_DTYPE}")

    llm = T5EncoderModel.from_pretrained(model_name, torch_dtype=MODEL_DTYPE).to(device)
    print("Model loaded successfully.")
    llm.eval()

    # Define hookpoints based on model architecture
    num_layers = llm.config.num_layers
    print(f"Detected {num_layers} layers for model: {model_name}")
    hookpoints = [f"encoder.blocks.{i}.hook_resid_post" for i in range(1, num_layers + 1)]
    harvester = ActivationsHarvester(llms=[llm], hookpoints=hookpoints, cache_mode="no_cache")

    shard_files = sorted(input_dir.glob("shard_*/protein_data.tsv"))
    if not shard_files:
        shard_files = sorted(input_dir.glob("shard_*.csv"))
    if not shard_files:
        shard_files = sorted(input_dir.glob("*.csv"))
    if not shard_files:
        shard_files = sorted(input_dir.glob("*.fasta"))
        if shard_files:
            print("Found .fasta files. Note: These will be treated as independent sequences without annotations.")

    print(f"Found {len(shard_files)} annotation files to process")

    for shard_file in tqdm(shard_files, desc="Processing shards"):
        if shard_file.suffix == ".fasta":
            sequences = []
            protein_ids = []
            print(f"Reading sequences from {shard_file.name}...")
            with open(shard_file) as f:
                for line in f:
                    if line.startswith(">"):
                        if max_sequences is not None and len(sequences) >= max_sequences:
                            break
                        protein_ids.append(line[1:].strip())
                        sequences.append("")
                    elif sequences:
                        sequences[-1] += line.strip()
            seq_col = "sequence"
        else:
            df = pd.read_csv(shard_file, sep="\t") if shard_file.suffix == ".tsv" else pd.read_csv(shard_file)

            seq_col = next((col for col in df.columns if col.lower() == sequence_column.lower()), None)
            if seq_col is None:
                raise ValueError(f"Column '{sequence_column}' not found in {shard_file}.")

            sequences = df[seq_col].tolist()
            protein_ids = (
                df[seq_col].index.tolist()
                if df.index.name
                else df["Entry"].tolist()
                if "Entry" in df.columns
                else list(range(len(sequences)))
            )

        if max_sequences is not None and shard_file.suffix != ".fasta":
            sequences = sequences[:max_sequences]
            protein_ids = protein_ids[:max_sequences]
            print(f"Limiting to first {max_sequences} sequences (TSV/CSV).")

        all_embeddings = []
        boundaries = []
        current_offset = 0

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]

            processed_seqs = [" ".join(list(seq)) for seq in batch_seqs]

            encoded = tokenizer(
                processed_seqs,
                add_special_tokens=True,
                padding=True,
                return_tensors="pt",
            )

            if i % (batch_size * 5) == 0:
                print(f"  Processed {i}/{len(sequences)} sequences...")

            input_ids = encoded["input_ids"].to(device)

            with torch.no_grad():
                acts_HSMPD = harvester.get_activations_HSMPD(input_ids)
            acts_HSMPD = acts_HSMPD.cpu()

            for b_idx in range(len(batch_seqs)):
                seq_len = len(batch_seqs[b_idx])
                if b_idx < acts_HSMPD.shape[0]:
                    valid_acts = acts_HSMPD[b_idx, 0:seq_len]  # Shape: [L, M, P, D]
                    all_embeddings.append(valid_acts)

                    boundaries.append((current_offset, current_offset + seq_len))
                    current_offset += seq_len

        cat_embeddings = torch.cat(all_embeddings, dim=0)

        if "shard_" in str(shard_file.parent.name):
            shard_dir = output_dir / shard_file.parent.name
            shard_dir.mkdir(parents=True, exist_ok=True)
            output_file = shard_dir / "embeddings.pt"
        else:
            output_file = output_dir / f"{shard_file.stem}.pt"

        save_data = {
            "embeddings": cat_embeddings,
            "boundaries": boundaries,
            "protein_ids": protein_ids,
        }
        torch.save(save_data, output_file)
        print(f"Saved embeddings with boundaries to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name", type=str, default="Rostlab/prot_t5_xl_uniref50")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--max_sequences", type=int, default=None, help="Limit number of sequences processed (for smoke test)"
    )
    args = parser.parse_args()

    embed_annotations(args.input_dir, args.output_dir, args.model_name, args.batch_size, args.max_sequences)
