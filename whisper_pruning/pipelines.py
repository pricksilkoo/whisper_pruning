from __future__ import annotations

from dataclasses import dataclass

from utils.WandA_profiler import WAprofiler
from utils.pruning_tools import PruningTool
from utils.scorer import Scorer

from .config import (
    EvaluationRunConfig,
    OneShotPruningRunConfig,
    OwlSweepConfig,
    ProfileRunConfig,
)
from .runtime import build_dataloader, build_evaluator, load_model_and_processor


@dataclass(slots=True)
class EvaluationResult:
    """评测最终返回的三个核心指标。"""

    cer: float
    wer: float
    avg_loss: float


@dataclass(slots=True)
class ProfileResult:
    """
    profiler 的输出结果。

    - weights: 当前模型每个线性层的权重
    - stats:   每个线性层对应的激活统计量
    """

    weights: dict
    stats: dict


@dataclass(slots=True)
class OneShotPruningResult:
    """
    一次性剪枝实验的完整输出。

    之所以把这些结果放在一个对象里，是为了后续你无论是打印、
    画图还是保存日志，都可以统一拿这个对象来处理。
    """

    evaluation: EvaluationResult
    scores: dict
    retention_ratio: dict
    sparsity: dict | float


def run_profile(config: ProfileRunConfig) -> ProfileResult:
    """
    只做一件事:
    读取模型 + 读取数据 + 跑 WAprofiler，得到 weights 和 stats。
    """
    model, processor, device, dtype = load_model_and_processor(config.experiment)
    dataloader = build_dataloader(config.data, processor, config.experiment)
    profiler = WAprofiler(model, dataloader, device=device, dtype=dtype)
    weights, stats = profiler.getWA()
    return ProfileResult(weights=weights, stats=stats)


def run_evaluation(config: EvaluationRunConfig) -> EvaluationResult:
    """
    只做模型评测。

    流程是:
    1. 加载模型和 processor
    2. 构造 dataloader
    3. 构造 evaluator
    4. 跑 evaluate()
    """
    model, processor, device, dtype = load_model_and_processor(config.experiment)
    dataloader = build_dataloader(config.data, processor, config.experiment)
    evaluator = build_evaluator(model, processor, dataloader, config.experiment, device, dtype)
    cer, wer, avg_loss = evaluator.evaluate()
    return EvaluationResult(cer=cer, wer=wer, avg_loss=avg_loss)


def _build_sparsity(scoring_result: dict, fallback: float | None = None) -> dict | float:
    """
    把“保留率 retention_ratio”转换成“稀疏度 sparsity”。

    举例:
    某层保留率是 0.4，意思是保留 40%，那稀疏度就是 0.6。
    """
    if scoring_result:
        return {name: 1.0 - ratio for name, ratio in scoring_result.items()}
    if fallback is None:
        raise ValueError("未提供 retention ratio，也没有提供 uniform_sparsity。")
    return fallback


def run_one_shot_pruning(config: OneShotPruningRunConfig) -> OneShotPruningResult:
    """
    这就是你项目里最核心的“一次性剪枝”流水线。

    按顺序做了 6 件事:
    1. 加载模型和 processor
    2. 用 profile_data 收集激活统计
    3. 用 scorer 计算每层分数 / 保留率
    4. 用 pruner 按保留率真正剪掉权重
    5. 用 eval_data 做推理评测
    6. 返回评测结果 + 中间分数
    """
    model, processor, device, dtype = load_model_and_processor(config.experiment)

    # 第一阶段: 用一小部分数据收集激活统计。
    profile_loader = build_dataloader(config.profile_data, processor, config.experiment)
    profiler = WAprofiler(model, profile_loader, device=device, dtype=dtype)
    weights, stats = profiler.getWA()

    # 第二阶段: 根据 weights 和 activations 计算“每层该保留多少”。
    scorer = Scorer()
    scores, retention_ratio = scorer.compute(
        method=config.pruning.scoring.method,
        weights=weights,
        activations_stats=stats,
        level=config.pruning.scoring.level,
        relative_difference=config.pruning.scoring.relative_difference,
        average_retention_ratio=config.pruning.scoring.average_retention_ratio,
    )

    # 第三阶段: 把 retention_ratio 变成 pruner 真正需要的 sparsity。
    sparsity = _build_sparsity(retention_ratio, fallback=config.pruning.uniform_sparsity)

    # 第四阶段: 执行剪枝。
    pruner = PruningTool()
    if config.pruning.method == "wanda_unstructured":
        pruner.wanda_unstructured_pruning(weights, stats, sparsity=sparsity)
    elif config.pruning.method == "wanda_nm":
        pruner.wanda_nm_pruning(weights, stats, n=config.pruning.n, m=config.pruning.m)
    else:
        raise ValueError(f"不支持的剪枝方法: {config.pruning.method}")

    # 第五阶段: 把剪枝后的权重写回模型。
    pruner.apply_to_model(model)

    # 第六阶段: 用测试集评估剪枝后的模型。
    eval_loader = build_dataloader(config.eval_data, processor, config.experiment)
    evaluator = build_evaluator(model, processor, eval_loader, config.experiment, device, dtype)
    cer, wer, avg_loss = evaluator.evaluate()

    return OneShotPruningResult(
        evaluation=EvaluationResult(cer=cer, wer=wer, avg_loss=avg_loss),
        scores=scores,
        retention_ratio=retention_ratio,
        sparsity=sparsity,
    )


def run_owl_sweep(config: OwlSweepConfig):
    """
    这个函数用于“扫参数”。

    它会反复尝试多组:
    - level
    - relative_difference
    - average_retention_ratio

    每组都执行一次“剪枝 + 评测”，最后把结果放进一个 4 维数组里。
    """
    import numpy as np

    model, processor, device, dtype = load_model_and_processor(config.experiment)

    # 先收集一次激活值统计，后面所有参数组合都复用这份统计。
    profile_loader = build_dataloader(config.profile_data, processor, config.experiment)
    profiler = WAprofiler(model, profile_loader, device=device, dtype=dtype)
    weights, stats = profiler.getWA()

    # 因为 sweep 过程中会反复剪枝，所以先备份原始权重，
    # 每跑完一组参数就恢复一次。
    original_weights_backup = {key: value.clone() for key, value in model.state_dict().items()}

    eval_loader = build_dataloader(config.eval_data, processor, config.experiment)
    evaluator = build_evaluator(model, processor, eval_loader, config.experiment, device, dtype)
    scorer = Scorer()

    results = np.zeros(
        (
            len(config.levels),
            len(config.relative_differences),
            len(config.average_retention_ratios),
            3,
        )
    )

    # 三重循环 = 枚举所有参数组合。
    for level_index, level in enumerate(config.levels):
        for difference_index, relative_difference in enumerate(config.relative_differences):
            for ratio_index, average_retention_ratio in enumerate(
                config.average_retention_ratios
            ):
                # 先根据当前参数算保留率。
                _, retention_ratio = scorer.owl(
                    weights=weights,
                    activations_stats=stats,
                    level=level,
                    relative_difference=relative_difference,
                    average_retention_ratio=average_retention_ratio,
                )

                sparsity = {name: 1.0 - score for name, score in retention_ratio.items()}
                pruner = PruningTool()
                pruner.wanda_unstructured_pruning(weights, stats, sparsity=sparsity)
                pruner.apply_to_model(model, log=False)

                # 评测结果放到 results 的最后一维:
                # [..., 0] = CER, [..., 1] = WER, [..., 2] = avg_loss
                cer, wer, avg_loss = evaluator.evaluate(log=False)
                results[level_index, difference_index, ratio_index] = [cer, wer, avg_loss]

                # 恢复原始权重，准备测试下一组参数。
                model.load_state_dict(original_weights_backup)

    return results
