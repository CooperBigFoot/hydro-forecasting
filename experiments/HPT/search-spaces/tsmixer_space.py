from typing import Any


def get_search_space() -> dict[str, dict[str, Any]]:
    """
    Define the hyperparameter search space for TSMixer models.

    Returns:
        Dictionary containing common and model-specific hyperparameter ranges
    """
    return {
        "common": {
            "input_length": {"type": "int", "low": 30, "high": 365},
            "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True},
        },
        "model_specific": {
            "hidden_size": {"type": "int", "low": 32, "high": 128},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "num_mixing_layers": {"type": "int", "low": 2, "high": 15},
            "static_embedding_size": {"type": "int", "low": 5, "high": 20},
            "fusion_method": {"type": "categorical", "choices": ["add", "concat"]},
        },
    }
