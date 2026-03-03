import torch
from pathlib import Path
from typing import Union, Optional

from crosscode.interplm_adapter.crosscoder_dictionary import CrosscoderDictionaryWrapper
from crosscode.models.acausal_crosscoder import ModelHookpointAcausalCrosscoder


def load_sae(
    model_dir: Union[str, Path], model_name: str = "ae.pt", device: Optional[str] = None
) -> CrosscoderDictionaryWrapper:
    """
    Load a pretrained ModelHookpointAcausalCrosscoder model in inference mode
    wrapped in a CrosscoderDictionaryWrapper to be used with InterPLM.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = Path(model_dir)

    model_path = model_dir / model_name
    crosscoder = ModelHookpointAcausalCrosscoder.load(model_path, device=device)
    crosscoder.eval()

    return CrosscoderDictionaryWrapper(crosscoder)
