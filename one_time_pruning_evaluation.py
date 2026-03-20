"""
这个文件对应你的核心实验：
1. 收集激活统计
2. 计算每层保留率
3. 执行一次性剪枝
4. 评测剪枝后的模型

以后你主要改下面的配置块，然后直接运行：
    python one_time_pruning_evaluation.py
"""

from experiment_helpers import load_data, load_model_and_processor
from utils.WandA_profiler import WAprofiler
from utils.evaluator import Evaluator
from utils.pruning_tools import PruningTool
from utils.scorer import Scorer


# ============================================================
# 只改这里
# ============================================================
MODEL_NAME = "whisper-large-v3-original"
DATASET_NAME = "en"
DTYPE = "float32"
DEVICE = None

MODEL_ROOT = "./models"
DATA_ROOT = "./data/fleurs_full"

# 用来收集激活值的小样本数据
PROFILE_SPLIT = "train"
PROFILE_BATCH_SIZE = 32
PROFILE_NUM_SAMPLES = 64

# 用来最终评测的数据
EVAL_SPLIT = "test"
EVAL_BATCH_SIZE = 16
EVAL_NUM_SAMPLES = 512

# 打分参数
SCORE_METHOD = "owl"
LEVEL = 7
RELATIVE_DIFFERENCE = 0
AVERAGE_RETENTION_RATIO = 0.4

# 剪枝参数
PRUNING_METHOD = "wanda_unstructured"  # 可选: wanda_unstructured / wanda_nm
UNIFORM_SPARSITY = None
N = 2
M = 4
# ============================================================


def main():
    model, processor, device, torch_dtype = load_model_and_processor(
        model_name=MODEL_NAME,
        dtype=DTYPE,
        device=DEVICE,
        model_root=MODEL_ROOT,
    )

    # 第一步：收集激活统计
    profile_loader = load_data(
        dataset_name=DATASET_NAME,
        processor=processor,
        split=PROFILE_SPLIT,
        batch_size=PROFILE_BATCH_SIZE,
        num_samples=PROFILE_NUM_SAMPLES,
        data_root=DATA_ROOT,
        shuffle=True,
    )

    profiler = WAprofiler(model, profile_loader, device=device, dtype=torch_dtype)
    weights, stats = profiler.getWA()

    # 第二步：根据分数算法计算每层保留率
    scorer = Scorer()
    scores, retention_ratio = scorer.compute(
        method=SCORE_METHOD,
        weights=weights,
        activations_stats=stats,
        level=LEVEL,
        relative_difference=RELATIVE_DIFFERENCE,
        average_retention_ratio=AVERAGE_RETENTION_RATIO,
    )

    # 如果打分方法没有返回 retention_ratio，就退回统一稀疏度模式
    if retention_ratio:
        sparsity = {name: 1.0 - ratio for name, ratio in retention_ratio.items()}
    elif UNIFORM_SPARSITY is not None:
        sparsity = UNIFORM_SPARSITY
    else:
        raise ValueError("当前配置没有得到 retention_ratio，也没有设置 UNIFORM_SPARSITY。")

    print(f"💻 打分完成，共得到 {len(scores)} 层分数")

    # 第三步：执行剪枝
    pruner = PruningTool()
    if PRUNING_METHOD == "wanda_unstructured":
        pruner.wanda_unstructured_pruning(weights, stats, sparsity=sparsity)
    elif PRUNING_METHOD == "wanda_nm":
        pruner.wanda_nm_pruning(weights, stats, n=N, m=M)
    else:
        raise ValueError(f"不支持的剪枝方法: {PRUNING_METHOD}")

    pruner.apply_to_model(model)

    # 第四步：评测剪枝后的模型
    eval_loader = load_data(
        dataset_name=DATASET_NAME,
        processor=processor,
        split=EVAL_SPLIT,
        batch_size=EVAL_BATCH_SIZE,
        num_samples=EVAL_NUM_SAMPLES,
        data_root=DATA_ROOT,
        shuffle=False,
    )

    evaluator = Evaluator(
        model=model,
        processor=processor,
        dataloader=eval_loader,
        device=device,
        language=DATASET_NAME,
        task="transcribe",
        dtype=torch_dtype,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()
