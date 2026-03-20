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
    cer: float
    wer: float
    avg_loss: float


@dataclass(slots=True)
class ProfileResult:
    weights: dict
    stats: dict


@dataclass(slots=True)
class OneShotPruningResult:
    evaluation: EvaluationResult
    scores: dict
    retention_ratio: dict
    sparsity: dict | float


def run_profile(config: ProfileRunConfig) -> ProfileResult:
    model, processor, device, dtype = load_model_and_processor(config.experiment)
    dataloader = build_dataloader(config.data, processor, config.experiment)
    profiler = WAprofiler(model, dataloader, device=device, dtype=dtype)
    weights, stats = profiler.getWA()
    return ProfileResult(weights=weights, stats=stats)


def run_evaluation(config: EvaluationRunConfig) -> EvaluationResult:
    model, processor, device, dtype = load_model_and_processor(config.experiment)
    dataloader = build_dataloader(config.data, processor, config.experiment)
    evaluator = build_evaluator(model, processor, dataloader, config.experiment, device, dtype)
    cer, wer, avg_loss = evaluator.evaluate()
    return EvaluationResult(cer=cer, wer=wer, avg_loss=avg_loss)


def _build_sparsity(scoring_result: dict, fallback: float | None = None) -> dict | float:
    if scoring_result:
        return {name: 1.0 - ratio for name, ratio in scoring_result.items()}
    if fallback is None:
        raise ValueError("未提供 retention ratio，也没有提供 uniform_sparsity。")
    return fallback


def run_one_shot_pruning(config: OneShotPruningRunConfig) -> OneShotPruningResult:
    model, processor, device, dtype = load_model_and_processor(config.experiment)

    profile_loader = build_dataloader(config.profile_data, processor, config.experiment)
    profiler = WAprofiler(model, profile_loader, device=device, dtype=dtype)
    weights, stats = profiler.getWA()

    scorer = Scorer()
    scores, retention_ratio = scorer.compute(
        method=config.pruning.scoring.method,
        weights=weights,
        activations_stats=stats,
        level=config.pruning.scoring.level,
        relative_difference=config.pruning.scoring.relative_difference,
        average_retention_ratio=config.pruning.scoring.average_retention_ratio,
    )

    sparsity = _build_sparsity(retention_ratio, fallback=config.pruning.uniform_sparsity)

    pruner = PruningTool()
    if config.pruning.method == "wanda_unstructured":
        pruner.wanda_unstructured_pruning(weights, stats, sparsity=sparsity)
    elif config.pruning.method == "wanda_nm":
        pruner.wanda_nm_pruning(weights, stats, n=config.pruning.n, m=config.pruning.m)
    else:
        raise ValueError(f"不支持的剪枝方法: {config.pruning.method}")

    pruner.apply_to_model(model)

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
    import numpy as np

    model, processor, device, dtype = load_model_and_processor(config.experiment)

    profile_loader = build_dataloader(config.profile_data, processor, config.experiment)
    profiler = WAprofiler(model, profile_loader, device=device, dtype=dtype)
    weights, stats = profiler.getWA()
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

    for level_index, level in enumerate(config.levels):
        for difference_index, relative_difference in enumerate(config.relative_differences):
            for ratio_index, average_retention_ratio in enumerate(
                config.average_retention_ratios
            ):
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

                cer, wer, avg_loss = evaluator.evaluate(log=False)
                results[level_index, difference_index, ratio_index] = [cer, wer, avg_loss]
                model.load_state_dict(original_weights_backup)

    return results
