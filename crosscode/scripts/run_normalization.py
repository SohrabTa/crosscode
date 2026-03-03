from interplm.sae import normalize
from crosscode.interplm_adapter.load_sae import load_sae

# Monkey-patch load_sae
normalize.load_sae = load_sae

if __name__ == "__main__":
    from tap import tapify

    tapify(normalize.normalize_sae_features)
