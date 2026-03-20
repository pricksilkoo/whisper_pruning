from whisper_pruning.config import DataLoaderConfig, ExperimentConfig, ProfileRunConfig
from whisper_pruning.pipelines import run_profile
from whisper_pruning.plotting import visualize_distributions


if __name__ == "__main__":
    model_name = "whisper-large-v3-original"
    dataset_name = "en"
    save_dir = f"./outputs/visualize_distributions/{model_name}"

    profile = run_profile(
        ProfileRunConfig(
            experiment=ExperimentConfig(
                model_name=model_name,
                dataset_name=dataset_name,
                dtype="float32",
            ),
            data=DataLoaderConfig(
                split="test",
                batch_size=4,
                num_samples=8,
                shuffle=False,
            ),
        )
    )

    visualize_distributions(profile.weights, profile.stats, save_dir, model_name)
