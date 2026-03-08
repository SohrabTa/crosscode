import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from crosscode.scripts.harvest_annotated_activations import embed_annotations


class TestHarvestActivations(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_dir = Path(self.temp_dir.name) / "input"
        self.output_dir = Path(self.temp_dir.name) / "output"

        self.input_dir.mkdir()
        self.output_dir.mkdir()

        df = pd.DataFrame({"sequence": ["MKTLLL", "MAPK"], "Entry": ["P12345", "P67890"]})

        shard_dir = self.input_dir / "shard_0"
        shard_dir.mkdir()
        df.to_csv(shard_dir / "protein_data.tsv", sep="\t", index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_harvesting_script(self):
        try:
            embed_annotations(
                self.input_dir,
                self.output_dir,
                model_name="t5-small",
                batch_size=2,
            )
        except Exception as e:
            self.fail(f"embed_annotations failed with {e}")

        output_file = self.output_dir / "shard_0" / "embeddings.pt"
        self.assertTrue(output_file.exists(), "Output file should be created")

        data = torch.load(output_file, weights_only=True)
        self.assertIn("embeddings", data)
        self.assertIn("boundaries", data)
        self.assertIn("protein_ids", data)

        embeddings = data["embeddings"]
        self.assertEqual(embeddings.shape[0], 10, "Should have exactly 10 valid tokens")
        self.assertEqual(len(data["boundaries"]), 2)
        self.assertEqual(data["boundaries"][0], (0, 6))
        self.assertEqual(data["boundaries"][1], (6, 10))
        self.assertEqual(data["protein_ids"], ["P12345", "P67890"])

        self.assertIsInstance(embeddings, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
