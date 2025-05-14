from typing import Any


def get_search_space() -> dict[str, dict[str, Any]]:
    """
    Define the hyperparameter search space for Entity-Aware LSTM models.

    Returns:
        Dictionary containing common and model-specific hyperparameter ranges
    """
    return {
        "common": {
            "input_length": {"type": "int", "low": 30, "high": 365},
            "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True},
        },
        "model_specific": {
            "num_layers": {"type": "int", "low": 1, "high": 3},
            "hidden_size": {"type": "int", "low": 32, "high": 256},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        },
    }
