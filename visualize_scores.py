from utils.scorer import Scorer
from whisper_pruning.config import DataLoaderConfig, ExperimentConfig, ProfileRunConfig
from whisper_pruning.pipelines import run_profile
from whisper_pruning.plotting import visualize_network_scores

# ==========================================
if __name__ == "__main__":
    model_name = "whisper-medium-original"
    dataset_name = "en"
    save_dir = f"./outputs/visualize_scores/{model_name}"
    score_method = "owl"
    filename = f"{score_method}_retention.png"

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

    scorer = Scorer()
    _, retention_ratios = scorer.compute(
        method=score_method,
        weights=profile.weights,
        activations_stats=profile.stats,
    )

    visualize_network_scores(
        retention_ratios,
        save_dir,
        filename,
        title=f"{score_method} retention across Whisper",
    )
