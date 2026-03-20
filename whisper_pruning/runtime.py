from __future__ import annotations

from typing import Optional

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

from utils.data_loader import get_whisper_dataloader
from utils.evaluator import Evaluator

from .config import DataLoaderConfig, ExperimentConfig


DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    try:
        return DTYPE_MAP[dtype_name.lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(DTYPE_MAP))
        raise ValueError(f"不支持的 dtype: {dtype_name}，可选值: {supported}") from exc


def resolve_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_text_normalizer(language: Optional[str]):
    if language and language.lower().startswith("en"):
        return EnglishTextNormalizer()
    return BasicTextNormalizer()


def load_model_and_processor(
    experiment: ExperimentConfig,
) -> tuple[WhisperForConditionalGeneration, WhisperProcessor, torch.device, torch.dtype]:
    dtype = resolve_torch_dtype(experiment.dtype)
    device = resolve_device(experiment.device)

    model = WhisperForConditionalGeneration.from_pretrained(
        str(experiment.model_path),
        torch_dtype=dtype,
    ).to(device)
    processor = WhisperProcessor.from_pretrained(str(experiment.model_path))

    return model, processor, device, dtype


def build_dataloader(
    data_config: DataLoaderConfig,
    processor: WhisperProcessor,
    experiment: ExperimentConfig,
):
    return get_whisper_dataloader(
        data_path=str(experiment.dataset_path),
        processor=processor,
        split=data_config.split,
        batch_size=data_config.batch_size,
        num_samples=data_config.num_samples,
        shuffle=data_config.should_shuffle,
    )


def build_evaluator(
    model,
    processor,
    dataloader,
    experiment: ExperimentConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> Evaluator:
    return Evaluator(
        model=model,
        processor=processor,
        dataloader=dataloader,
        device=device,
        language=experiment.language,
        task=experiment.task,
        dtype=dtype,
        normalizer=get_text_normalizer(experiment.language),
    )
