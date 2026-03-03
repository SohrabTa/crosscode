from interplm.analysis.concepts import compare_activations
from crosscode.interplm_adapter.load_sae import load_sae

# Monkey-patch load_sae
compare_activations.load_sae = load_sae

if __name__ == "__main__":
    from tap import tapify

    tapify(compare_activations.analyze_all_shards_in_set)
