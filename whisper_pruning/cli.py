from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    DataLoaderConfig,
    EvaluationRunConfig,
    ExperimentConfig,
    OwlSweepConfig,
    OneShotPruningRunConfig,
    PruningConfig,
    ProfileRunConfig,
    ProjectPaths,
    ScoringConfig,
)


def _project_paths(args) -> ProjectPaths:
    return ProjectPaths(
        model_root=Path(args.models_root) if args.models_root else None,
        data_root=Path(args.data_root) if args.data_root else None,
        output_root=Path(args.output_root) if args.output_root else None,
    )


def _experiment_config(args) -> ExperimentConfig:
    return ExperimentConfig(
        model_name=args.model,
        dataset_name=args.dataset,
        dtype=args.dtype,
        device=args.device,
        language=args.language,
        task=args.task,
        paths=_project_paths(args),
    )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="whisper-large-v3-original")
    parser.add_argument("--dataset", default="en")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--models-root", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Whisper pruning experiment runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="只做模型评测")
    _add_common_args(evaluate_parser)
    evaluate_parser.add_argument("--split", default="test")
    evaluate_parser.add_argument("--batch-size", type=int, default=64)
    evaluate_parser.add_argument("--num-samples", type=int, default=None)
    evaluate_parser.add_argument("--shuffle", action="store_true")

    prune_parser = subparsers.add_parser("prune-once", help="执行一次剪枝并评测")
    _add_common_args(prune_parser)
    prune_parser.add_argument("--profile-split", default="train")
    prune_parser.add_argument("--profile-batch-size", type=int, default=32)
    prune_parser.add_argument("--profile-num-samples", type=int, default=64)
    prune_parser.add_argument("--eval-split", default="test")
    prune_parser.add_argument("--eval-batch-size", type=int, default=16)
    prune_parser.add_argument("--eval-num-samples", type=int, default=512)
    prune_parser.add_argument("--score-method", default="owl", choices=["owl", "hhr1", "cv", "mean"])
    prune_parser.add_argument("--level", type=float, default=7.0)
    prune_parser.add_argument("--relative-difference", type=float, default=0.0)
    prune_parser.add_argument("--average-retention-ratio", type=float, default=0.4)
    prune_parser.add_argument("--pruning-method", default="wanda_unstructured", choices=["wanda_unstructured", "wanda_nm"])
    prune_parser.add_argument("--uniform-sparsity", type=float, default=None)
    prune_parser.add_argument("--n", type=int, default=2)
    prune_parser.add_argument("--m", type=int, default=4)

    score_parser = subparsers.add_parser("plot-scores", help="可视化层分数")
    _add_common_args(score_parser)
    score_parser.add_argument("--split", default="test")
    score_parser.add_argument("--batch-size", type=int, default=4)
    score_parser.add_argument("--num-samples", type=int, default=8)
    score_parser.add_argument("--score-method", default="owl", choices=["owl", "hhr1", "cv", "mean"])
    score_parser.add_argument("--level", type=float, default=7.0)
    score_parser.add_argument("--relative-difference", type=float, default=0.0)
    score_parser.add_argument("--average-retention-ratio", type=float, default=0.4)
    score_parser.add_argument("--plot", default="retention", choices=["scores", "retention"])
    score_parser.add_argument("--filename", default=None)

    distribution_parser = subparsers.add_parser("plot-distributions", help="可视化层分布")
    _add_common_args(distribution_parser)
    distribution_parser.add_argument("--split", default="test")
    distribution_parser.add_argument("--batch-size", type=int, default=4)
    distribution_parser.add_argument("--num-samples", type=int, default=8)
    distribution_parser.add_argument("--layer", default=None)

    sweep_parser = subparsers.add_parser("sweep-owl", help="扫描 OWL 参数并画图")
    _add_common_args(sweep_parser)
    sweep_parser.add_argument("--profile-split", default="train")
    sweep_parser.add_argument("--profile-batch-size", type=int, default=64)
    sweep_parser.add_argument("--profile-num-samples", type=int, default=64)
    sweep_parser.add_argument("--eval-split", default="test")
    sweep_parser.add_argument("--eval-batch-size", type=int, default=32)
    sweep_parser.add_argument("--eval-num-samples", type=int, default=None)
    sweep_parser.add_argument("--levels", type=float, nargs="+", default=[8.0, 9.0])
    sweep_parser.add_argument(
        "--relative-differences",
        type=float,
        nargs="+",
        default=[step * 0.03 for step in range(11)],
    )
    sweep_parser.add_argument(
        "--average-retention-ratios",
        type=float,
        nargs="+",
        default=[step * 0.05 for step in range(6, 9)],
    )

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    experiment = _experiment_config(args)

    if args.command == "evaluate":
        from .pipelines import run_evaluation

        config = EvaluationRunConfig(
            experiment=experiment,
            data=DataLoaderConfig(
                split=args.split,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                shuffle=args.shuffle,
            ),
        )
        result = run_evaluation(config)
        print(f"评测完成: CER={result.cer:.2%}, WER={result.wer:.2%}, Loss={result.avg_loss:.4f}")
        return

    if args.command == "prune-once":
        from .pipelines import run_one_shot_pruning

        config = OneShotPruningRunConfig(
            experiment=experiment,
            profile_data=DataLoaderConfig(
                split=args.profile_split,
                batch_size=args.profile_batch_size,
                num_samples=args.profile_num_samples,
            ),
            eval_data=DataLoaderConfig(
                split=args.eval_split,
                batch_size=args.eval_batch_size,
                num_samples=args.eval_num_samples,
                shuffle=False,
            ),
            pruning=PruningConfig(
                method=args.pruning_method,
                scoring=ScoringConfig(
                    method=args.score_method,
                    level=args.level,
                    relative_difference=args.relative_difference,
                    average_retention_ratio=args.average_retention_ratio,
                ),
                uniform_sparsity=args.uniform_sparsity,
                n=args.n,
                m=args.m,
            ),
        )
        result = run_one_shot_pruning(config)
        print(
            "剪枝评测完成: "
            f"CER={result.evaluation.cer:.2%}, "
            f"WER={result.evaluation.wer:.2%}, "
            f"Loss={result.evaluation.avg_loss:.4f}"
        )
        return

    if args.command == "plot-scores":
        from utils.scorer import Scorer

        from .pipelines import run_profile
        from .plotting import visualize_network_scores

        config = ProfileRunConfig(
            experiment=experiment,
            data=DataLoaderConfig(
                split=args.split,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                shuffle=False,
            ),
        )
        profile = run_profile(config)
        scorer = Scorer()
        scores, retention_ratio = scorer.compute(
            method=args.score_method,
            weights=profile.weights,
            activations_stats=profile.stats,
            level=args.level,
            relative_difference=args.relative_difference,
            average_retention_ratio=args.average_retention_ratio,
        )
        values = retention_ratio if args.plot == "retention" else scores
        output_dir = experiment.paths.output_root / "visualize_scores" / experiment.model_name
        filename = args.filename or f"{args.score_method}_{args.plot}.png"
        visualize_network_scores(
            values,
            save_dir=str(output_dir),
            filename=filename,
            title=f"{args.score_method} {args.plot} across Whisper",
        )
        return

    if args.command == "plot-distributions":
        from .pipelines import run_profile
        from .plotting import visualize_distributions

        config = ProfileRunConfig(
            experiment=experiment,
            data=DataLoaderConfig(
                split=args.split,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                shuffle=False,
            ),
        )
        profile = run_profile(config)
        output_dir = experiment.paths.output_root / "visualize_distributions" / experiment.model_name
        visualize_distributions(
            profile.weights,
            profile.stats,
            save_dir=str(output_dir),
            model_name=experiment.model_name,
            selected_layer=args.layer,
        )
        return

    if args.command == "sweep-owl":
        from .pipelines import run_owl_sweep
        from .plotting import plot_fixed_arr, plot_fixed_level

        config = OwlSweepConfig(
            experiment=experiment,
            profile_data=DataLoaderConfig(
                split=args.profile_split,
                batch_size=args.profile_batch_size,
                num_samples=args.profile_num_samples,
            ),
            eval_data=DataLoaderConfig(
                split=args.eval_split,
                batch_size=args.eval_batch_size,
                num_samples=args.eval_num_samples,
                shuffle=False,
            ),
            levels=args.levels,
            relative_differences=args.relative_differences,
            average_retention_ratios=args.average_retention_ratios,
        )
        results = run_owl_sweep(config)
        for level in config.levels:
            plot_fixed_level(
                target_level=level,
                levels=config.levels,
                relative_differences=config.relative_differences,
                average_retention_ratios=config.average_retention_ratios,
                results=results,
                save_dir=str(config.output_dir),
            )
        for average_retention_ratio in config.average_retention_ratios:
            plot_fixed_arr(
                target_arr=average_retention_ratio,
                levels=config.levels,
                relative_differences=config.relative_differences,
                average_retention_ratios=config.average_retention_ratios,
                results=results,
                save_dir=str(config.output_dir),
            )
        return

    raise ValueError(f"未知命令: {args.command}")
