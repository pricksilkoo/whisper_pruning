import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import WhisperForConditionalGeneration, WhisperProcessor

models = {
    "openai/whisper-base": "../models/whisper-base-original",
    "openai/whisper-medium": "../models/whisper-medium-original",
    "openai/whisper-large-v3": "../models/whisper-large-v3-original"
}

for model_id, local_dir in models.items():
    print(f"⏳ 正在拉取模型: {model_id} ...")
    
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    processor = WhisperProcessor.from_pretrained(model_id)

    os.makedirs(local_dir, exist_ok=True)
    
    model.save_pretrained(local_dir)
    processor.save_pretrained(local_dir)
    
    print(f"✅ 成功！{model_id} 已保存至 {local_dir}\n")

print("🎉 所有模型下载完毕！")