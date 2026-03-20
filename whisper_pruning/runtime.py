from __future__ import annotations

from typing import Optional

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

from utils.data_loader import get_whisper_dataloader
from utils.evaluator import Evaluator

from .config import DataLoaderConfig, ExperimentConfig


# 把字符串 dtype 映射成 PyTorch 认识的 torch.dtype。
# 这样你在配置里写 "float16"，底层就能变成 torch.float16。
DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    """把配置里的 dtype 字符串转换成 torch.dtype。"""
    try:
        return DTYPE_MAP[dtype_name.lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(DTYPE_MAP))
        raise ValueError(f"不支持的 dtype: {dtype_name}，可选值: {supported}") from exc


def resolve_device(preferred: Optional[str] = None) -> torch.device:
    """
    决定模型最终放在哪个设备上。

    优先级:
    1. 用户手动指定的 device
    2. CUDA
    3. MPS
    4. CPU
    """
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_text_normalizer(language: Optional[str]):
    """
    评测前需要把预测文本和真实文本先做标准化。

    英文用 EnglishTextNormalizer，
    其他语言先退回到更通用的 BasicTextNormalizer。
    """
    if language and language.lower().startswith("en"):
        return EnglishTextNormalizer()
    return BasicTextNormalizer()


def load_model_and_processor(
    experiment: ExperimentConfig,
) -> tuple[WhisperForConditionalGeneration, WhisperProcessor, torch.device, torch.dtype]:
    """
    根据 ExperimentConfig 一次性完成:
    - dtype 决定
    - device 决定
    - model 加载
    - processor 加载

    这样外部就不用在每个脚本里重复写同一套加载逻辑。
    """
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
    """
    用统一方式创建 dataloader。

    输入是两个配置对象:
    - experiment: 告诉我们数据集路径在哪
    - data_config: 告诉我们 split / batch_size / num_samples
    """
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
    """
    统一构造 Evaluator。

    这样评测逻辑的公共参数也都集中在一个地方处理，
    不需要每个脚本都手工 new Evaluator。
    """
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
