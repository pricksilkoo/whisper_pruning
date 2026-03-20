import matplotlib.pyplot as plt
import numpy as np
import re
import os
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from utils.WandA_profiler import WAprofiler
from utils.scorer import Scorer
from utils.data_loader import get_whisper_dataloader 

def visualize_network_scores(scores, save_dir="score_plots", filename="network_scores.png", title="Layer-wise Scores across the Network"):
    """
    按模型连接顺序可视化各层分数
    """
    if not scores:
        print("❌ 传入的 scores 字典为空！")
        return

    # Whisper 拓扑排序规则
    def whisper_sort_key(layer_name):
        if "model.encoder" in layer_name or ".encoder." in layer_name:
            module_priority = 0
        elif "model.decoder" in layer_name or ".decoder." in layer_name:
            module_priority = 1
        else:
            module_priority = 2
        match = re.search(r'layers\.(\d+)', layer_name)
        layer_num = int(match.group(1)) if match else -1

        if "self_attn.q_proj" in layer_name: sub_order = 1
        elif "self_attn.k_proj" in layer_name: sub_order = 2
        elif "self_attn.v_proj" in layer_name: sub_order = 3
        elif "self_attn.out_proj" in layer_name: sub_order = 4
        elif "encoder_attn.q_proj" in layer_name: sub_order = 5
        elif "encoder_attn.k_proj" in layer_name: sub_order = 6
        elif "encoder_attn.v_proj" in layer_name: sub_order = 7
        elif "encoder_attn.out_proj" in layer_name: sub_order = 8
        elif "fc1" in layer_name: sub_order = 9
        elif "fc2" in layer_name: sub_order = 10
        else: sub_order = 99
        
        sub_components = [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', layer_name)]
        
        return (module_priority, layer_num, sub_order, sub_components)

    sorted_layer_names = sorted(scores.keys(), key=whisper_sort_key)
    sorted_scores = [scores[name] for name in sorted_layer_names]

    encoder_decoder_boundary = None
    layer_boundaries = []
    
    current_module = "encoder" if "model.encoder" in sorted_layer_names[0] else None
    current_layer_num = re.search(r'layers\.(\d+)', sorted_layer_names[0])
    current_layer_num = int(current_layer_num.group(1)) if current_layer_num else -1

    for i, name in enumerate(sorted_layer_names):
        is_decoder = "model.decoder" in name or ".decoder." in name
        if is_decoder and current_module == "encoder":
            encoder_decoder_boundary = i
            current_module = "decoder"
            
        match = re.search(r'layers\.(\d+)', name)
        l_num = int(match.group(1)) if match else -1
        if l_num != current_layer_num and l_num != -1: 
            layer_boundaries.append(i)
            current_layer_num = l_num

    #绘图
    plt.figure(figsize=(24, 6))
    x_indices = np.arange(len(sorted_scores))
    
    plt.plot(x_indices, sorted_scores, marker='o', markersize=4, linestyle='-', linewidth=1.5, color='dodgerblue', alpha=0.8, zorder=3)
    plt.fill_between(x_indices, sorted_scores, 0, color='dodgerblue', alpha=0.1)

    for boundary in layer_boundaries:
        plt.axvline(x=boundary - 0.5, color='gray', linestyle=':', linewidth=1, alpha=0.6, zorder=1)

    if encoder_decoder_boundary is not None:
        plt.axvline(x=encoder_decoder_boundary - 0.5, color='red', linestyle='-', linewidth=2.5, alpha=0.9, zorder=2, label="Encoder-Decoder Boundary")

    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.ylabel("Score Value", fontsize=14)
    plt.xlabel("Layer Index (Topology Order)", fontsize=14) 
    plt.xlim(-1, len(sorted_layer_names))
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    if encoder_decoder_boundary is not None:
        plt.legend(fontsize=12, loc="upper right")

    step = max(1, len(sorted_layer_names) // 40) 
    plt.xticks(x_indices[::step], x_indices[::step], fontsize=10)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 拓扑分数图已成功保存至: {os.path.abspath(save_path)}")
    plt.close()

# ==========================================
if __name__ == "__main__":
    #配置路径
    model_name = "whisper-medium-original" # 选择model里面的模型
    dataset_name = "en"  # 选择fleurs_full里面的语言
    model_path = f"./models/{model_name}" 
    data_path = f"./data/fleurs_full/{dataset_name}"
    save_dir = f"./data/visualize_scores/{model_name}" #选择保存路径
    score_method = "OWL_retention_ratio" #修改后记得还有主函数里面方法修改！！！这个关系文件命名！！！
    filename = f"{score_method}_scores.png"
    dtype=torch.float32 #模型使用精度

    #数据获取
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype).to(device)
    processor = WhisperProcessor.from_pretrained(model_path)
    dataloader = get_whisper_dataloader(
        batch_size=4, 
        data_path=data_path, 
        num_samples=8, #可以修改
        processor=processor,
        split="test" 
    )
    profiler = WAprofiler(model, dataloader, device, dtype)
    weights, stats = profiler.getWA()

    #分数获取
    scorer = Scorer()
    scores, retention_ratios = scorer.owl(weights, stats) #自行选择打分方法！！！

    # 运行画图函数，注意修改所看指标！！！
    visualize_network_scores(retention_ratios, save_dir, filename, title=f"{score_method} Scores Across Whisper")