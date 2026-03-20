import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" #指定卡跑
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from utils.evaluator import Evaluator 
from utils.data_loader import get_whisper_dataloader
from utils.WandA_profiler import WAprofiler
from utils.pruning_tools import PruningTool
from utils.scorer import Scorer

device = "cuda" if torch.cuda.is_available() else "cpu"

#配置路径
model_name = "whisper-large-v3-original" 
dataset_name = "en"
model_path = f"./models/{model_name}" 
data_path = f"./data/fleurs_full/{dataset_name}"
torch_dtype=torch.float32 #模型使用精度

# 加载模型
print(f"🚀 正在加载模型: {model_name} ")
model = WhisperForConditionalGeneration.from_pretrained(
    model_path, 
    torch_dtype=torch_dtype
).to(device)
processor = WhisperProcessor.from_pretrained(model_path)

# 加载数据集
print(f"📂 正在加载数据集: {dataset_name}")
dataloader = get_whisper_dataloader(
    batch_size=32,        # A40单卡推理largeFP16最大size64
    data_path=data_path, 
    num_samples=64,      
    processor=processor,
    split="train"        
)

# 获取weights和stats（activations）
print(f"💼正在获取权重和激活值")
profiler = WAprofiler(model, dataloader, device)
weights, stats = profiler.getWA()

# 用相关算法获取每层的分数（稀疏度）
print(f"💻正在计算非均匀稀疏度")
scorer=Scorer()
_,retention_ratio=scorer.owl(weights,stats,level=7,relative_difference=0,average_retention_ratio=0.4)

# 按照稀疏度进行一次性剪枝
print(f"✂️正在进行一次性剪枝")
sparsity_dict = {name: 1.0 - score for name, score in retention_ratio.items()}
pruner = PruningTool()
pruned_W, masks = pruner.wanda_unstructured_pruning(weights,stats,sparsity=sparsity_dict)
pruner.apply_to_model(model)

dataloader = get_whisper_dataloader(
    batch_size=16,        # A40单卡推理largeFP16最大size64
    data_path=data_path, 
    num_samples=512,       # 可以设为None     
    processor=processor,
    split="test"        
)

evaluator_after = Evaluator(model, processor, dataloader, device)
evaluator_after.evaluate()