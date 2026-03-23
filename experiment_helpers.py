"""
这个文件只放少量“重复但简单”的辅助函数。

你平时主要修改这些顶层脚本就够了：
- eval.py
- prune_once.py
- plot_scores.py
- plot_distributions.py
- sweep_owl.py

通常不需要改这个文件。
"""

import os

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from utils.dataloader import get_whisper_dataloader


DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def configure_torch_runtime():
    """
    为推理场景打开一些相对稳妥的加速开关。
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def get_torch_dtype(dtype_name="float16"):
    """把字符串 dtype 转成 PyTorch 能识别的 dtype。"""
    if isinstance(dtype_name, torch.dtype):
        return dtype_name

    dtype_name = dtype_name.lower()
    if dtype_name not in DTYPE_MAP:
        raise ValueError(f"不支持的 dtype: {dtype_name}")
    return DTYPE_MAP[dtype_name]


def get_device(device=None):
    """
    自动决定设备。

    - 如果你手动传了 device，就优先用它
    - 否则优先 CUDA
    - 再否则 MPS
    - 最后 CPU
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model_path(model_name, model_root="./models"):
    model_name = os.path.expanduser(str(model_name))
    if os.path.isabs(model_name) or os.path.exists(model_name) or os.path.sep in model_name:
        return model_name
    return os.path.join(model_root, model_name)


def get_data_path(dataset_name, data_root="./data/fleurs_full"):
    return os.path.join(data_root, dataset_name)


def load_model_and_processor(
    model_name,
    dtype="float16",
    device=None,
    model_root="./models",
):
    """
    加载 Whisper 模型和 processor。

    返回:
    - model
    - processor
    - device
    - torch_dtype
    """
    torch_dtype = get_torch_dtype(dtype)
    device = get_device(device)
    model_path = get_model_path(model_name, model_root=model_root)

    if os.path.isfile(model_path):
        checkpoint_path = model_path
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
        if state_dict is None:
            state_dict = checkpoint

        base_model_name = checkpoint_config.get("model_name")
        if not base_model_name:
            raise ValueError(
                "剪枝 checkpoint 缺少 `config.model_name`，无法定位原始 Whisper 模型。"
            )

        base_model_path = get_model_path(base_model_name, model_root=model_root)
        print(f"🚀 正在加载原始模型骨架: {base_model_path}")
        print(f"📦 正在加载剪枝 checkpoint: {checkpoint_path}")
        model = WhisperForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
        ).to(device)
        model.load_state_dict(state_dict)
        processor = WhisperProcessor.from_pretrained(base_model_path)
        return model, processor, device, torch_dtype

    print(f"🚀 正在加载模型: {model_path}")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
    ).to(device)
    processor = WhisperProcessor.from_pretrained(model_path)

    return model, processor, device, torch_dtype


def load_data(
    dataset_name,
    processor,
    split="test",
    batch_size=4,
    num_samples=None,
    data_root="./data/fleurs_full",
    shuffle=None,
    random_subset=False,
    seed=42,
    num_workers=0,
    pin_memory=None,
    shard_id=None,
    num_shards=None,
    text_field="raw_transcription",
):
    """
    加载一个 Whisper dataloader。

    你最常改的就是:
    - split
    - batch_size
    - num_samples
    """
    data_path = get_data_path(dataset_name, data_root=data_root)
    print(f"📂 正在加载数据集: {data_path} [{split}]")

    return get_whisper_dataloader(
        data_path=data_path,
        processor=processor,
        split=split,
        batch_size=batch_size,
        num_samples=num_samples,
        shuffle=shuffle,
        random_subset=random_subset,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shard_id=shard_id,
        num_shards=num_shards,
        text_field=text_field,
    )
