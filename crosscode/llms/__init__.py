from typing import cast

import torch
from transformer_lens import HookedTransformer  # type: ignore
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    T5EncoderModel,
    T5Tokenizer,
)

from crosscode.log import logger
from crosscode.saveable_module import STRING_TO_DTYPE
from crosscode.trainers.config_common import LLMConfig


def build_llms(
    llms: list[LLMConfig],
    cache_dir: str,
    device: torch.device,
    inferenced_type: str,
) -> list[HookedTransformer | T5EncoderModel]:
    return [build_llm(llm, cache_dir, device, inferenced_type)[0] for llm in llms]


def build_llm(
    llm: LLMConfig,
    cache_dir: str,
    device: torch.device,
    inference_dtype: str,
) -> tuple[HookedTransformer | T5EncoderModel, PreTrainedTokenizerBase]:
    dtype = STRING_TO_DTYPE[inference_dtype]

    if llm.name is not None:
        model_key = f"tl-{llm.name}"
        if llm.revision:
            model_key += f"_rev-{llm.revision}"

        llm_out = HookedTransformer.from_pretrained_no_processing(
            llm.name,
            revision=llm.revision,
            cache_dir=cache_dir,
            device=device,
            dtype=dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            llm.name,
            revision=llm.revision,
            cache_dir=cache_dir,
        )
    else:
        assert llm.base_architecture_name is not None
        assert llm.hf_model_name is not None

        model_key = f"tl-{llm.base_architecture_name}_hf-{llm.hf_model_name}"

        if "prot_t5" in llm.hf_model_name.lower():
            logger.info(f"Loading HuggingFace model {llm.hf_model_name} into T5EncoderModel")
            llm_out = T5EncoderModel.from_pretrained(
                llm.hf_model_name,
                cache_dir=cache_dir,
                torch_dtype=dtype,
            )
            tokenizer = T5Tokenizer.from_pretrained(
                llm.hf_model_name,
                do_lower_case=False,
                cache_dir=cache_dir,
            )
        else:
            logger.info(
                f"Loading HuggingFace model {llm.hf_model_name} into HookedTransformer model {llm.base_architecture_name}"
            )
            llm_out = HookedTransformer.from_pretrained_no_processing(
                llm.base_architecture_name,
                hf_model=AutoModelForCausalLM.from_pretrained(llm.hf_model_name, cache_dir=cache_dir),
                cache_dir=cache_dir,
                device=device,
                dtype=dtype,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                llm.hf_model_name,
                cache_dir=cache_dir,
            )

    # Replace any slashes with underscores to avoid potential path issues
    model_key = model_key.replace("/", "_").replace("\\", "_")

    if isinstance(llm_out, T5EncoderModel):
        # Attach tokenizer for use in build_model_hookpoint_dataloader
        llm_out.tokenizer = tokenizer
    else:
        # Register the model key as a buffer so it's properly accessible
        # Buffers are persistent state in nn.Module that's not parameters
        llm_out.register_buffer("crosscode_model_key", torch.tensor([ord(c) for c in model_key], dtype=torch.int64))

    logger.info(f"Assigned model key: {model_key}")

    return cast(HookedTransformer | T5EncoderModel, llm_out.to(device)), tokenizer
