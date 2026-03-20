import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_from_disk
from transformers import WhisperProcessor

#处理原始数据
def prepare_dataset(batch,processor):

    audio=batch["audio"]
    inputs = processor.feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=True  
    )
    batch["input_features"] = inputs.input_features[0]

    if "attention_mask" in inputs:
        batch["attention_mask"] = inputs.attention_mask[0]

    batch["labels"]=processor.tokenizer(batch["raw_transcription"]).input_ids

    return batch

#数据补齐
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor:Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        
        input_features = []
        for feature in features:
            feat_dict = {"input_features": feature["input_features"]}
            if "attention_mask" in feature:
                feat_dict["attention_mask"] = feature["attention_mask"]
            input_features.append(feat_dict)
            
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        
        return batch

#数据加载器，对外api
def get_whisper_dataloader(data_path, processor, split="train", batch_size=4, num_samples=None):
    """
    参数说明:
    - data_path: 本地数据的路径
    - processor: WhisperProcessor
    - split: "train", "validation" 或 "test"
    - batch_size: 一次送进去几句话
    - num_samples: 截取数量(None为全部)
    """
    dataset = load_from_disk(data_path)[split]

    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    dataset = dataset.map(
        lambda x: prepare_dataset(x, processor),
        remove_columns=dataset.column_names, 
        num_proc=1, # 核
        desc="提取特征"
    )
    dataset.cleanup_cache_files()

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=(split == "train")
    )

    return dataloader