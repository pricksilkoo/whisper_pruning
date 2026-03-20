import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from utils.WandA_profiler import WAprofiler
from utils.pruning_tools import PruningTool
from utils.scorer import Scorer
from utils.evaluator import Evaluator 
from utils.data_loader import get_whisper_dataloader 

def one_time_owl_pruning(model, processor, dataloader, scorer, weights, stats, 
                         level, relative_difference, average_retention_ratio, 
                         device, dtype=torch.float16 ):

    _,retention_ratio=scorer.owl(weights,stats,level,relative_difference,average_retention_ratio)
    sparsity_dict = {name: 1.0 - score for name, score in retention_ratio.items()}
    pruner = PruningTool()
    pruner.wanda_unstructured_pruning(weights,stats,sparsity=sparsity_dict)
    pruner.apply_to_model(model,log=False)
    evaluator_after = Evaluator(model, processor, dataloader, device, dtype=dtype)
    cer, wer, avg_loss = evaluator_after.evaluate(log=False)

    return cer, wer, avg_loss

def plot_fixed_level(target_level, levels, relative_differences, average_retention_ratios, f, save_dir):
    # 1. 找到目标 level 在列表中的序号 (x轴索引)
    x_idx = levels.index(target_level)
    
    # 2. 创建一张宽 15，高 5 的大图，里面包含 1 行 3 列的子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics_names = ['CER', 'WER', 'Avg Loss']
    
    # 3. 遍历所有的 average_retention_ratio (z轴索引)
    for z_idx, arr in enumerate(average_retention_ratios):
        # 取出当前 arr 下，所有 relative_differences 的 3 个指标的值
        # 切片含义：f[固定的x, 所有的y, 当前的z, 所有的指标]
        data_slice = f[x_idx, :, z_idx, :]  # shape: (len(relative_differences), 3)
        
        # 4. 在 3 个子图上分别画线
        for metric_idx in range(3):
            axes[metric_idx].plot(
                relative_differences,         # X 轴
                data_slice[:, metric_idx],    # Y 轴 (取出对应的指标)
                marker='o',                   # 给数据点加个圆圈标记
                label=f'ARR = {arr}'          # 图例名字
            )
            
    # 5. 设置图表格式
    for i, ax in enumerate(axes):
        ax.set_title(f'{metrics_names[i]} (Level = {target_level})')
        ax.set_xlabel('Relative Difference')
        ax.set_ylabel(metrics_names[i])
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend() # 显示图例

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"fixed_level_{target_level}_owl.png")    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 成功保存对比图至: {save_path}")

# 使用示例：
# plot_fixed_level(5, levels, relative_differences, average_retention_ratios, f)
    
def plot_fixed_arr(target_arr, levels, relative_differences, average_retention_ratios, f, save_dir):
    # 1. 找到目标 average_retention_ratio 的序号 (z轴索引)
    z_idx = average_retention_ratios.index(target_arr)
    
    # 2. 创建 1 行 3 列的子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics_names = ['CER', 'WER', 'Avg Loss']
    
    # 3. 遍历所有的 level (x轴索引)
    for x_idx, lvl in enumerate(levels):
        # 切片含义：f[当前的x, 所有的y, 固定的z, 所有的指标]
        data_slice = f[x_idx, :, z_idx, :]  # shape: (len(relative_differences), 3)
        
        # 4. 画线
        for metric_idx in range(3):
            axes[metric_idx].plot(
                relative_differences, 
                data_slice[:, metric_idx], 
                marker='s',                 # 换个方形标记区分一下
                label=f'Level = {lvl}'
            )
            
    # 5. 设置图表格式
    for i, ax in enumerate(axes):
        ax.set_title(f'{metrics_names[i]} (ARR = {target_arr})')
        ax.set_xlabel('Relative Difference')
        ax.set_ylabel(metrics_names[i])
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"fixed_arr_{target_arr}_owl.png")    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 成功保存对比图至: {save_path}")

# 使用示例：
# plot_fixed_arr(0.6, levels, relative_differences, average_retention_ratios, f)

# ==========================================
if __name__ == "__main__":

    #配置路径
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "whisper-large-v3-original" # 选择model里面的模型
    dataset_name = "en"  # 选择fleurs_full里面的语言
    model_path = f"./models/{model_name}" 
    data_path = f"./data/fleurs_full/{dataset_name}"
    save_dir = f"./data/visualize_owl_onetimepruning/{model_name}" #选择保存路径
    dtype=torch.float16 #模型使用精度

    #设置变量范围
    levels=[8,9]
    relative_differences=[x * 0.03 for x in range(0, 11)] 
    average_retention_ratios=[x * 0.05 for x in range(6, 9)]
    
    # 加载模型
    print(f"🚀 正在加载模型: {model_name} ")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=dtype
    ).to(device)
    processor = WhisperProcessor.from_pretrained(model_path)

    # 加载数据集，用于获取激活值
    print(f"📂 正在加载数据集: {dataset_name}")
    dataloader = get_whisper_dataloader(
        batch_size=64,        # A40单卡推理largeFP16最大size64
        data_path=data_path, 
        num_samples=64,      
        processor=processor,
        split="train"        
    )

    # 获取weights和stats（activations）,以及原始权重
    print(f"💼正在获取权重和激活值")
    profiler = WAprofiler(model, dataloader, device)
    weights, stats = profiler.getWA()
    original_weights_backup = {k: v.clone() for k, v in model.state_dict().items()}

    # 加载推理评估使用的数据集
    dataloader = get_whisper_dataloader(
        batch_size=32,        # A40单卡推理largeFP16最大size64
        data_path=data_path, 
        num_samples=None,       # 可以设为None     
        processor=processor,
        split="test"        
    )

    #打分器初始化
    scorer=Scorer()

    #进行评估
    f = np.zeros((len(levels), len(relative_differences), len(average_retention_ratios), 3))
    for x,level in enumerate(levels):
        for y,relative_difference in enumerate(relative_differences):
            for z,average_retention_ratio in enumerate(average_retention_ratios):
                cer, wer, avg_loss = one_time_owl_pruning(model, processor, dataloader, scorer, weights,
                                    stats, level, relative_difference, average_retention_ratio,
                                    device, dtype)
                f[x,y,z]=[cer, wer, avg_loss]
                model.load_state_dict(original_weights_backup)

    for level in levels:
        plot_fixed_level(target_level=level,levels=levels,relative_differences=relative_differences,
                        average_retention_ratios=average_retention_ratios,f=f,save_dir=save_dir)  
    for average_retention_ratio in average_retention_ratios:
        plot_fixed_arr(target_arr=average_retention_ratio,levels=levels,relative_differences=relative_differences,
                        average_retention_ratios=average_retention_ratios,f=f,save_dir=save_dir)          
