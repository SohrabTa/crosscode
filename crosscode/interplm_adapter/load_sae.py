from pathlib import Path

from crosscode.interplm_adapter.crosscoder_dictionary import CrosscoderDictionaryWrapper
from crosscode.models.acausal_crosscoder import ModelHookpointAcausalCrosscoder
from crosscode.utils import get_device


def load_sae(
    model_dir: str | Path, model_name: str = "ae.pt", device: str | None = None
) -> CrosscoderDictionaryWrapper:
    """
    Load a pretrained ModelHookpointAcausalCrosscoder model in inference mode
    wrapped in a CrosscoderDictionaryWrapper to be used with InterPLM.
    """
    if device is None:
        device = get_device()
    model_dir = Path(model_dir)

    model_path = model_dir / model_name
    crosscoder = ModelHookpointAcausalCrosscoder.load(model_path, device=device)
    crosscoder.eval()

    return CrosscoderDictionaryWrapper(crosscoder)
