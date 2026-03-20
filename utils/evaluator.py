#单卡推理
import torch
from tqdm import tqdm
import jiwer
import whisper.normalizers as Normalizers

class Evaluator:
    def __init__(self, model, processor, dataloader, device, 
                 language='en', task="transcribe", dtype=torch.float32,
                 normalizer=Normalizers.EnglishTextNormalizer()):
        self.model = model
        self.processor = processor
        self.dataloader = dataloader
        self.device = device
        self.normalizer = normalizer
        self.task=task
        self.language=language
        self.dtype=dtype

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
                #向前传播
                outputs = self.model(input_features=input_features, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                #复用 encoder_outputs
                encoder_outputs = outputs.encoder_last_hidden_state
                #自回归推理
                generated_ids = self.model.generate(
                    input_features=input_features,
                    encoder_outputs=(encoder_outputs,), 
                    attention_mask=attention_mask,
                    language=self.language,     
                    task=self.task,             
                )
                transcriptions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                predictions.extend(transcriptions)
                #文本对比
                labels_for_decode = labels.clone()
                labels_for_decode[labels_for_decode == -100] = self.processor.tokenizer.pad_token_id
                real_texts = self.processor.batch_decode(labels_for_decode, skip_special_tokens=True)
                references.extend(real_texts)

        clean_references = [self.normalizer(text) for text in references]
        clean_predictions = [self.normalizer(text) for text in predictions]
        # 计算cer，wer，avgloss
        cer = jiwer.cer(clean_references, clean_predictions)
        wer = jiwer.wer(clean_references, clean_predictions)
        avg_loss = total_loss / len(self.dataloader)
        
        #打印日志
        if (len(predictions) and log) > 0:
            print(f" 测评结果 | Loss: {avg_loss:.4f} | CER (字符错误率): {cer:.2%} | WER (词错误率): {wer:.2%}")
            print("\n 【样例抽查】:")
            for i in range(5):
                print(f"真实文本: {references[i]}")
                print(f"模型预测: {predictions[i]}")
                print("="*100)
        
        return cer, wer, avg_loss