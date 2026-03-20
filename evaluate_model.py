from whisper_pruning.config import DataLoaderConfig, EvaluationRunConfig, ExperimentConfig
from whisper_pruning.pipelines import run_evaluation


if __name__ == "__main__":
    config = EvaluationRunConfig(
        experiment=ExperimentConfig(
            model_name="whisper-large-v3-original",
            dataset_name="en",
            dtype="float16",
        ),
        data=DataLoaderConfig(
            split="test",
            batch_size=64,
            num_samples=None,
            shuffle=False,
        ),
    )

    run_evaluation(config)
