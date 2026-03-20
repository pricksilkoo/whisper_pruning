"""
这个文件对应你的核心实验：
1. 收集激活统计
2. 计算剪枝稀疏度
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
PROFILE_NUM_SAMPLES = 256
PROFILE_RANDOM_SUBSET = True
PROFILE_SAMPLE_SEED = 42

# 用来最终评测的数据
EVAL_SPLIT = "test"
EVAL_BATCH_SIZE = 16
EVAL_NUM_SAMPLES = 512

# 剪枝模式
# - "uniform": 真正的统一剪枝，所有层都用同一个 sparsity
# - "layerwise": 先打分，再给不同层分配不同 sparsity
SPARSITY_MODE = "uniform"  # uniform / layerwise
UNIFORM_SPARSITY = 0.6

# 只有在 SPARSITY_MODE="layerwise" 时，这组参数才会生效
SCORE_METHOD = "owl"  # owl / cv
LEVEL = 7
RELATIVE_DIFFERENCE = 0
AVERAGE_RETENTION_RATIO = 0.4

# ============================================================


def build_sparsity(weights, stats):
    """
    根据配置决定这次到底是:
    - 真 uniform 剪枝
    - 还是 layerwise 剪枝
    """
    if SPARSITY_MODE == "uniform":
        print(f"✂️ 使用 uniform Wanda，统一稀疏度 = {UNIFORM_SPARSITY:.2%}")
        return UNIFORM_SPARSITY

    if SPARSITY_MODE == "layerwise":
        scorer = Scorer()
        scores, retention_ratio = scorer.compute(
            method=SCORE_METHOD,
            weights=weights,
            activations_stats=stats,
            level=LEVEL,
            relative_difference=RELATIVE_DIFFERENCE,
            average_retention_ratio=AVERAGE_RETENTION_RATIO,
        )
        sparsity = {name: 1.0 - ratio for name, ratio in retention_ratio.items()}
        print(f"💻 {SCORE_METHOD} 打分完成，共得到 {len(scores)} 层分数")
        return sparsity

    raise ValueError(f"不支持的 SPARSITY_MODE: {SPARSITY_MODE}")


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
        random_subset=PROFILE_RANDOM_SUBSET,
        seed=PROFILE_SAMPLE_SEED,
    )

    profiler = WAprofiler(model, profile_loader, device=device, dtype=torch_dtype)
    weights, stats = profiler.getWA()

    # 第二步：生成这次要用的 sparsity
    sparsity = build_sparsity(weights, stats)

    # 第三步：执行剪枝
    pruner = PruningTool()
    pruner.wanda_unstructured_pruning(weights, stats, sparsity=sparsity)
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
