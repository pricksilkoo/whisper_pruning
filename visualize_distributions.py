import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from utils.WandA_profiler import WAprofiler
from utils.data_loader import get_whisper_dataloader 


def to_numpy_flat(data):
    """辅助函数：将可能传入的 PyTorch Tensor 或 List 转为展平的 1D NumPy 数组"""
    if hasattr(data, 'detach'):
        data = data.detach().cpu().numpy()
    return np.array(data).flatten()

def visualize_distributions(weights, stats, save_dir, model_name):
    """
    交互式可视化指定层的权重、激活值及其乘积分布，并保存为高分辨率图片
    """
    common_layers = sorted(list(set(weights.keys()) & set(stats.keys())))
    
    if not common_layers:
        print("❌ 错误：weights 和 stats 字典中没有匹配的层名！")
        return

    print("\n" + "="*50)
    print(" 📊 可用的网络层对照表 ")
    print("="*50)
    for i, layer_name in enumerate(common_layers):
        print(f"[{i:03d}] {layer_name}")
    print("="*50 + "\n")

    selected_layer = None
    while True:
        choice = input(f"请输入想要可视化的层编号 (0-{len(common_layers)-1})，或输入 'q' 退出: ")
        if choice.lower() == 'q':
            print("已退出可视化。")
            return
        
        try:
            idx = int(choice)
            if 0 <= idx < len(common_layers):
                selected_layer = common_layers[idx]
                break
            else:
                print("⚠️ 编号超出范围，请重新输入。")
        except ValueError:
            print("⚠️ 输入无效，请输入数字编号。")

    print(f"\n⏳ 正在处理并绘制 [{selected_layer}] 的数据，请稍候...")

    w_raw = weights[selected_layer]
    w_raw = torch.abs(w_raw)
    s_raw = (stats[selected_layer]["sq_sum"]/stats[selected_layer]["count"]).sqrt()
    try:
        prod_raw = w_raw * s_raw 
    except Exception as e:
        print(f"❌ 矩阵相乘失败 (维度不匹配? W: {w_raw.shape}, S: {s_raw.shape})\n错误信息: {e}")
        return

    w_data = to_numpy_flat(w_raw)
    s_data = to_numpy_flat(s_raw)
    prod_data = to_numpy_flat(prod_raw)

    w_mean = np.mean(w_data)
    s_mean = np.mean(s_data)
    prod_mean = np.mean(prod_data)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5)) 
    fig.suptitle(f'Distribution Analysis: {model_name}--{selected_layer}', fontsize=16, fontweight='bold')

    def plot_with_mean(ax, data, mean_val, title, color):
        sns.histplot(data, bins=100, kde=True, color=color, ax=ax, stat="probability")
        
        ax.set_title(f'{title}\nMean: {mean_val:.6f}', fontsize=13)
        ax.set_xlabel('Value')
        ax.set_ylabel('Probability (Log Scale)')
        ax.set_yscale('log')
        ax.set_ylim(1e-7, 1)
        
        ax.axvline(x=mean_val, color='red', linestyle='--', linewidth=2, alpha=0.8)

    plot_with_mean(axes[0], w_data, w_mean, 'Weights (|W|) Distribution', 'steelblue')
    plot_with_mean(axes[1], s_data, s_mean, 'Activations (||X||_l2) Distribution', 'darkorange')
    plot_with_mean(axes[2], prod_data, prod_mean, 'Product (|W| * ||X||_l2) Distribution', 'seagreen')

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    safe_layer_name = selected_layer.replace(".", "_").replace("/", "_")
    save_path = os.path.join(save_dir, f"dist_{safe_layer_name}.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图片已成功保存至: {os.path.abspath(save_path)}")
    
# ==========================================
if __name__ == "__main__":

    #配置路径
    model_name = "whisper-large-v3-original" # 选择model里面的模型
    dataset_name = "en"  # 选择fleurs_full里面的语言
    model_path = f"./models/{model_name}" 
    data_path = f"./data/fleurs_full/{dataset_name}"
    save_dir = f"./data/visualize_distributions/{model_name}" #选择保存路径
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

    visualize_distributions(weights, stats, save_dir, model_name)