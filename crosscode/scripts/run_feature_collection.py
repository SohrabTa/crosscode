from interplm.sae import inference

from crosscode.interplm_adapter.load_sae import load_sae

inference.load_sae = load_sae

if __name__ == "__main__":
    import runpy
    from pathlib import Path

    script_path = Path(__import__("interplm").__file__).parent.parent / "scripts" / "collect_feature_activations.py"

    runpy.run_path(str(script_path), run_name="__main__")
