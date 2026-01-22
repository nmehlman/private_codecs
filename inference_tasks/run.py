from inference_tasks.tasks import CategoricalEmotionInference, MaskedCategoricalEmotionInference
import pickle
import os
import argparse
import yaml

TASKS = {
    "categorical_emotion": CategoricalEmotionInference,
    "masked_categorical_emotion": MaskedCategoricalEmotionInference
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run inference for a specified task.")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file."
    )
    
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    inference_class = TASKS[config.pop("task")]
    save_path = config.pop("save_path", None)
    batch_size = config.pop("batch_size", 32)
    
    inference_instance = inference_class(**config)
    results = inference_instance.run(batch_size=batch_size)
    
    # Save or process results as needed
    print(f"Inference completed")
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Results saved to {save_path}")