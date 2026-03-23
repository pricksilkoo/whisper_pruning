#单卡推理
import torch
from tqdm import tqdm
import jiwer
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer


WHISPER_LANGUAGE_ALIASES = {
    "en": "english",
    "english": "english",
    "zh": "chinese",
    "chinese": "chinese",
    "cmn_hans_cn": "chinese",
    "ja": "japanese",
    "jp": "japanese",
    "japanese": "japanese",
    "ko": "korean",
    "kr": "korean",
    "korean": "korean",
    "fr": "french",
    "french": "french",
    "de": "german",
    "german": "german",
    "es": "spanish",
    "spanish": "spanish",
    "ru": "russian",
    "russian": "russian",
    "hi": "hindi",
    "hindi": "hindi",
    "ar": "arabic",
    "arabic": "arabic",
}


def resolve_whisper_language(language):
    """
    把数据集缩写或语言代码映射成 Whisper generate 更稳定的语言名。
    """
    if language is None:
        return None

    language = str(language).strip()
    if not language:
        return None

    if language.startswith("<|") and language.endswith("|>"):
        return language

    return WHISPER_LANGUAGE_ALIASES.get(language.lower(), language)


def get_text_normalizer(language):
    """
    根据语言选择文本归一化器。

    参数:
    - language: 语言名或语言代码，例如 `en`。
    """
    if language and language.lower().startswith("en"):
        return EnglishTextNormalizer()
    return BasicTextNormalizer()


def compute_metrics(references, predictions, normalizer):
    """
    计算 CER/WER。

    参数:
    - references: 参考文本列表。
    - predictions: 模型预测文本列表。
    - normalizer: 文本归一化函数或 normalizer 对象。
    """
    clean_references = [normalizer(text) for text in references]
    clean_predictions = [normalizer(text) for text in predictions]
    cer = jiwer.cer(clean_references, clean_predictions)
    wer = jiwer.wer(clean_references, clean_predictions)
    return cer, wer, clean_references, clean_predictions


def print_evaluation_summary(avg_loss, cer, wer, references, predictions, log=True):
    """
    打印评测摘要和若干条样例。

    参数:
    - avg_loss: 平均 loss。
    - cer: 字符错误率。
    - wer: 词错误率。
    - references: 原始参考文本列表。
    - predictions: 原始预测文本列表。
    - log: 是否打印日志。
    """
    if not predictions or not log:
        return

    loss_text = f"{avg_loss:.4f}" if avg_loss == avg_loss else "N/A"
    print(f" 测评结果 | Loss: {loss_text} | CER (字符错误率): {cer:.2%} | WER (词错误率): {wer:.2%}")
    print("\n 【样例抽查】:")
    for i in range(min(5, len(predictions))):
        print(f"真实文本: {references[i]}")
        print(f"模型预测: {predictions[i]}")
        print("=" * 100)

class Evaluator:
    """
    这个类负责最终评测模型。

    它会做两件事:
    1. 前向传播拿到训练式 loss
    2. generate 解码拿到真实转录结果，再计算 CER / WER
    """

    def __init__(self, model, processor, dataloader, device, 
                 language='en', task="transcribe", dtype=None,
                 normalizer=None, generation_kwargs=None, compute_loss=True):
        """
        参数:
        - model: Whisper 模型。
        - processor: WhisperProcessor。
        - dataloader: 评测数据 dataloader。
        - device: 推理设备。
        - language: generate 时使用的语言。
        - task: Whisper generate 的任务类型，通常是 `transcribe`。
        - dtype: 输入特征送入模型时使用的 dtype。
        - normalizer: 自定义文本归一化器；不传则按 language 自动选择。
        - generation_kwargs: 额外的 generate 参数字典。
        - compute_loss: 是否顺带计算前向 loss。
        """
        self.model = model
        self.processor = processor
        self.dataloader = dataloader
        self.device = device
        self.normalizer = normalizer or get_text_normalizer(language)
        self.task=task
        self.language=language
        self.whisper_language = resolve_whisper_language(language)
        self.dtype=dtype or next(model.parameters()).dtype
        generation_kwargs = dict(generation_kwargs or {})
        self.generation_batch_size = generation_kwargs.pop("generation_batch_size", None)
        self.generation_kwargs = generation_kwargs
        self.forced_decoder_ids = self.generation_kwargs.get("forced_decoder_ids")
        if self.forced_decoder_ids is None and self.whisper_language is not None:
            try:
                self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=self.whisper_language,
                    task=self.task,
                )
            except Exception:
                self.forced_decoder_ids = None
        self.compute_loss = compute_loss

    def _generate_in_chunks(self, input_features, attention_mask):
        """
        generate 比普通 forward 更吃显存，尤其是 beam search。
        所以这里支持把一个大 batch 拆成多个小块，逐块解码。

        参数:
        - input_features: 当前 batch 的输入特征张量。
        - attention_mask: 当前 batch 的 attention mask，可为 `None`。
        """
        batch_size = input_features.shape[0]
        chunk_size = self.generation_batch_size or batch_size
        chunk_size = max(1, min(chunk_size, batch_size))

        predictions = []
        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            chunk_input_features = input_features[start:end]
            chunk_attention_mask = None if attention_mask is None else attention_mask[start:end]

            generate_kwargs = {
                "input_features": chunk_input_features,
                "attention_mask": chunk_attention_mask,
            }
            generate_kwargs.update(self.generation_kwargs)
            if "forced_decoder_ids" not in generate_kwargs:
                if self.forced_decoder_ids is not None:
                    generate_kwargs["forced_decoder_ids"] = self.forced_decoder_ids
                else:
                    if self.whisper_language is not None:
                        generate_kwargs.setdefault("language", self.whisper_language)
                    if self.task is not None:
                        generate_kwargs.setdefault("task", self.task)

            generated_ids = self.model.generate(**generate_kwargs)
            chunk_predictions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            predictions.extend(chunk_predictions)

            del generated_ids, chunk_input_features
            if chunk_attention_mask is not None:
                del chunk_attention_mask
            if chunk_size < batch_size and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return predictions

    def evaluate(self, log = True, return_details=False):
        """
        运行完整的推理评估，计算 Loss (交叉熵)，计算 CER (字符错误率)， WER (词错误率)。

        参数:
        - log: 是否打印评测摘要。
        - return_details: 是否返回详细结果字典；否则只返回 `(cer, wer, avg_loss)`。
        """
        self.model.eval()
        
        predictions = []
        references = []
        total_loss = 0.0

        with torch.inference_mode():
            for batch in tqdm(self.dataloader, desc="正在进行语音转录推理"):
                input_features = batch["input_features"].to(self.device, dtype=self.dtype)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                if self.compute_loss:
                    # 前向时把 attention_mask 也传进去，避免 padding 对结果造成额外污染。
                    outputs = self.model(
                        input_features=input_features,
                        labels=labels,
                        attention_mask=attention_mask,
                    )
                    total_loss += outputs.loss.item()
                    del outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # 这里不再复用 forward 拿到的 encoder_outputs，
                # 直接让 generate 自己重新走一遍编码，优先保证评测正确。
                transcriptions = self._generate_in_chunks(
                    input_features=input_features,
                    attention_mask=attention_mask,
                )
                predictions.extend(transcriptions)

                # 评测时优先使用数据集原始参考文本，
                # 不再把 labels 反解码回字符串，避免 tokenizer 往返引入口径偏差。
                reference_texts = batch.get("reference_texts")
                if reference_texts is not None:
                    references.extend(reference_texts)
                else:
                    labels_for_decode = labels.clone()
                    labels_for_decode[labels_for_decode == -100] = self.processor.tokenizer.pad_token_id
                    real_texts = self.processor.batch_decode(labels_for_decode, skip_special_tokens=True)
                    references.extend(real_texts)

        cer, wer, clean_references, clean_predictions = compute_metrics(
            references=references,
            predictions=predictions,
            normalizer=self.normalizer,
        )
        avg_loss = total_loss / len(self.dataloader) if self.compute_loss else float("nan")

        print_evaluation_summary(avg_loss, cer, wer, references, predictions, log=log)

        if return_details:
            return {
                "cer": cer,
                "wer": wer,
                "avg_loss": avg_loss,
                "total_loss": total_loss,
                "num_batches": len(self.dataloader),
                "num_loss_batches": len(self.dataloader) if self.compute_loss else 0,
                "compute_loss": self.compute_loss,
                "references": references,
                "predictions": predictions,
                "clean_references": clean_references,
                "clean_predictions": clean_predictions,
            }

        return cer, wer, avg_loss
