#单卡推理
import torch
from tqdm import tqdm
import jiwer
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

class Evaluator:
    """
    这个类负责最终评测模型。

    它会做两件事:
    1. 前向传播拿到训练式 loss
    2. generate 解码拿到真实转录结果，再计算 CER / WER
    """

    def __init__(self, model, processor, dataloader, device, 
                 language='en', task="transcribe", dtype=None,
                 normalizer=None):
        self.model = model
        self.processor = processor
        self.dataloader = dataloader
        self.device = device
        self.normalizer = normalizer or self._build_normalizer(language)
        self.task=task
        self.language=language
        self.dtype=dtype or next(model.parameters()).dtype

    def _build_normalizer(self, language):
        # 英文用更强的英文标准化器，其他语言先用基础版。
        if language and language.lower().startswith("en"):
            return EnglishTextNormalizer()
        return BasicTextNormalizer()

    def evaluate(self, log = True):
        """
        运行完整的推理评估，计算 Loss (交叉熵)，计算 CER (字符错误率)， WER (词错误率)。
        """
        self.model.eval()
        
        predictions = []
        references = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="正在进行语音转录推理"):
                input_features = batch["input_features"].to(self.device, dtype=self.dtype)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                # 前向时把 attention_mask 也传进去，避免 padding 对结果造成额外污染。
                outputs = self.model(
                    input_features=input_features,
                    labels=labels,
                    attention_mask=attention_mask,
                )
                loss = outputs.loss
                total_loss += loss.item()

                # 这里不再复用 forward 拿到的 encoder_outputs，
                # 直接让 generate 自己重新走一遍编码，优先保证评测正确。
                generated_ids = self.model.generate(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    language=self.language,     
                    task=self.task,             
                )
                transcriptions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                predictions.extend(transcriptions)
                #文本对比

                # labels 里 padding 部分通常是 -100，不能直接拿去 decode，
                # 所以先替换成 pad_token_id。
                labels_for_decode = labels.clone()
                labels_for_decode[labels_for_decode == -100] = self.processor.tokenizer.pad_token_id
                real_texts = self.processor.batch_decode(labels_for_decode, skip_special_tokens=True)
                references.extend(real_texts)

        # 先做文本标准化，再计算误差率。
        clean_references = [self.normalizer(text) for text in references]
        clean_predictions = [self.normalizer(text) for text in predictions]
        # 计算cer，wer，avgloss
        cer = jiwer.cer(clean_references, clean_predictions)
        wer = jiwer.wer(clean_references, clean_predictions)
        avg_loss = total_loss / len(self.dataloader)
        
        #打印日志
        if predictions and log:
            print(f" 测评结果 | Loss: {avg_loss:.4f} | CER (字符错误率): {cer:.2%} | WER (词错误率): {wer:.2%}")
            print("\n 【样例抽查】:")
            for i in range(min(5, len(predictions))):
                print(f"真实文本: {references[i]}")
                print(f"模型预测: {predictions[i]}")
                print("="*100)
        
        return cer, wer, avg_loss
