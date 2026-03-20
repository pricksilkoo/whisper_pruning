import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" #指定卡跑
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from utils.evaluator import Evaluator 
from utils.data_loader import get_whisper_dataloader

device = "cuda" if torch.cuda.is_available() else "cpu"

# 配置部分
model_name = "whisper-large-v3-original" 
dataset_name = "en"
model_path = f"./models/{model_name}" 
data_path = f"./data/fleurs_full/{dataset_name}"
torch_dtype=torch.float16 #模型使用精度

# 加载模型
print(f"🚀 正在加载模型: {model_name} ")
model = WhisperForConditionalGeneration.from_pretrained(
    model_path, 
    torch_dtype=torch_dtype
).to(device)
processor = WhisperProcessor.from_pretrained(model_path)

#加载数据集
print(f"📂 正在加载数据集: {dataset_name}")
dataloader = get_whisper_dataloader(
    batch_size=64,        # A40单卡推理largeFP16最大size64
    data_path=data_path, 
    num_samples=None,      #数据量
    processor=processor,
    split="test"        
)

baseline_evaluator = Evaluator(
    model=model, 
    processor=processor, 
    dataloader=dataloader, 
    device=device,
    language=dataset_name,
    task="transcribe"
)

baseline_evaluator.evaluate()