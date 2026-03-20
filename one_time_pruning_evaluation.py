from whisper_pruning.config import (
    DataLoaderConfig,
    ExperimentConfig,
    OneShotPruningRunConfig,
    PruningConfig,
    ScoringConfig,
)
from whisper_pruning.pipelines import run_one_shot_pruning


if __name__ == "__main__":
    config = OneShotPruningRunConfig(
        experiment=ExperimentConfig(
            model_name="whisper-large-v3-original",
            dataset_name="en",
            dtype="float32",
        ),
        profile_data=DataLoaderConfig(
            split="train",
            batch_size=32,
            num_samples=64,
        ),
        eval_data=DataLoaderConfig(
            split="test",
            batch_size=16,
            num_samples=512,
            shuffle=False,
        ),
        pruning=PruningConfig(
            method="wanda_unstructured",
            scoring=ScoringConfig(
                method="owl",
                level=7,
                relative_difference=0,
                average_retention_ratio=0.4,
            ),
        ),
    )

    run_one_shot_pruning(config)
