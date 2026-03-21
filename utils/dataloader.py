import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from datasets import load_from_disk


def prepare_dataset(batch, processor, text_field="raw_transcription"):
    """
    把原始样本转成 Whisper 可直接使用的特征格式。

    参数:
    - batch: 数据集里的一条原始样本，至少包含 `audio` 和文本字段。
    - processor: WhisperProcessor，用来提取语音特征和编码文本标签。
    - text_field: 文本字段名，默认读取 `raw_transcription`。
    """
    audio = batch["audio"]
    inputs = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=True,
    )
    batch["input_features"] = inputs.input_features[0]

    if "attention_mask" in inputs:
        batch["attention_mask"] = inputs.attention_mask[0]

    if text_field not in batch:
        raise KeyError(f"数据里找不到文本字段: {text_field}")

    batch["reference_text"] = batch[text_field]
    batch["labels"] = processor.tokenizer(batch["reference_text"]).input_ids
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        """
        把一个样本列表补齐成 batch。

        参数:
        - features: `prepare_dataset` 处理后的样本列表。
        """
        input_features = []
        for feature in features:
            feature_dict = {"input_features": feature["input_features"]}
            if "attention_mask" in feature:
                feature_dict["attention_mask"] = feature["attention_mask"]
            input_features.append(feature_dict)

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        batch["reference_texts"] = [feature["reference_text"] for feature in features]
        return batch


def sample_dataset(dataset, num_samples: Optional[int], random_subset: bool, seed: int):
    """
    从完整数据集中截取子集。

    参数:
    - dataset: HuggingFace Dataset。
    - num_samples: 需要保留的样本数，`None` 表示全量。
    - random_subset: 是否先随机打乱再截取。
    - seed: 随机抽样时使用的随机种子。
    """
    if num_samples is None or num_samples >= len(dataset):
        return dataset

    if random_subset:
        dataset = dataset.shuffle(seed=seed)

    return dataset.select(range(num_samples))


def get_whisper_dataloader(
    data_path,
    processor,
    split="train",
    batch_size=4,
    num_samples=None,
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
    构建 Whisper 评测/分析用 dataloader。

    参数:
    - data_path: 本地 `load_from_disk` 数据目录。
    - processor: WhisperProcessor。
    - split: 使用的数据切分，如 `train/test/validation`。
    - batch_size: dataloader 的 batch size。
    - num_samples: 截取样本数，`None` 表示全量。
    - shuffle: 是否打乱；传 `None` 时仅训练集默认打乱。
    - random_subset: 截取前是否先随机抽样。
    - seed: 随机抽样种子。
    - num_workers: dataloader worker 数。
    - pin_memory: 是否启用 pin memory；传 `None` 时按 CUDA 自动决定。
    - shard_id: 当前分片编号，多卡评测时使用。
    - num_shards: 总分片数，多卡评测时使用。
    - text_field: 标签文本字段名。
    """
    dataset_dict = load_from_disk(data_path)
    dataset = dataset_dict[split]
    dataset = sample_dataset(
        dataset=dataset,
        num_samples=num_samples,
        random_subset=random_subset,
        seed=seed,
    )

    if shard_id is not None and num_shards is not None:
        dataset = dataset.shard(num_shards=num_shards, index=shard_id, contiguous=True)

    dataset = dataset.map(
        lambda batch: prepare_dataset(batch, processor, text_field=text_field),
        remove_columns=dataset.column_names,
        num_proc=1,
        desc="提取特征",
        keep_in_memory=True,
        load_from_cache_file=False,
    )

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=(split == "train") if shuffle is None else shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory,
        persistent_workers=num_workers > 0,
    )


__all__ = [
    "DataCollatorSpeechSeq2SeqWithPadding",
    "get_whisper_dataloader",
    "prepare_dataset",
    "sample_dataset",
]
