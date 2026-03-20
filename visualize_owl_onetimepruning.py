from whisper_pruning.config import DataLoaderConfig, ExperimentConfig, OwlSweepConfig
from whisper_pruning.pipelines import run_owl_sweep
from whisper_pruning.plotting import plot_fixed_arr, plot_fixed_level


if __name__ == "__main__":
    model_name = "whisper-large-v3-original"
    dataset_name = "en"
    save_dir = f"./outputs/visualize_owl_onetimepruning/{model_name}"

    levels = [8, 9]
    relative_differences = [x * 0.03 for x in range(0, 11)]
    average_retention_ratios = [x * 0.05 for x in range(6, 9)]

    config = OwlSweepConfig(
        experiment=ExperimentConfig(
            model_name=model_name,
            dataset_name=dataset_name,
            dtype="float16",
        ),
        profile_data=DataLoaderConfig(
            split="train",
            batch_size=64,
            num_samples=64,
        ),
        eval_data=DataLoaderConfig(
            split="test",
            batch_size=32,
            num_samples=None,
            shuffle=False,
        ),
        levels=levels,
        relative_differences=relative_differences,
        average_retention_ratios=average_retention_ratios,
        output_dir=save_dir,
    )

    results = run_owl_sweep(config)

    for level in levels:
        plot_fixed_level(
            target_level=level,
            levels=levels,
            relative_differences=relative_differences,
            average_retention_ratios=average_retention_ratios,
            results=results,
            save_dir=save_dir,
        )

    for average_retention_ratio in average_retention_ratios:
        plot_fixed_arr(
            target_arr=average_retention_ratio,
            levels=levels,
            relative_differences=relative_differences,
            average_retention_ratios=average_retention_ratios,
            results=results,
            save_dir=save_dir,
        )
