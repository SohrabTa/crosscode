import argparse
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
import os


def count_residues(fasta_path):
    if not os.path.exists(fasta_path):
        print(f"Error: File not found at {fasta_path}")
        return

    lengths = []
    total_residues = 0

    print(f"Reading FASTA file: {fasta_path}")

    # We use a simple loop to iterate through the records
    # Bio.SeqIO.parse is an iterator, so it's memory efficient
    for record in tqdm(SeqIO.parse(fasta_path, "fasta"), desc="Processing sequences"):
        seq_len = len(record.seq)
        lengths.append(seq_len)
        total_residues += seq_len

    num_sequences = len(lengths)

    if num_sequences == 0:
        print("No sequences found in the file.")
        return

    lengths = np.array(lengths)

    print("\n--- Statistics ---")
    print(f"Total Sequences: {num_sequences:,}")
    print(f"Total Residues:  {total_residues:,}")
    print(f"Mean Length:     {np.mean(lengths):.2f}")
    print(f"Median Length:   {np.median(lengths):.2f}")
    print(f"Min Length:      {np.min(lengths):.2f}")
    print(f"Max Length:      {np.max(lengths):.2f}")
    print(f"Std Dev Length:  {np.std(lengths):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count residues in a FASTA file.")
    parser.add_argument("fasta_path", type=str, help="Path to the FASTA file.")
    args = parser.parse_args()
    count_residues(args.fasta_path)
