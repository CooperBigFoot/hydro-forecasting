from typing import Any


def get_search_space() -> dict[str, dict[str, Any]]:
    """
    Define the hyperparameter search space for Temporal Fusion Transformer models.

    Returns:
        Dictionary containing common and model-specific hyperparameter ranges
    """
    return {
        "common": {
            "input_length": {"type": "int", "low": 30, "high": 365},
            # Make hidden_size always divisible by 8 (max num_attention_heads)
            "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True},
        },
        "model_specific": {
            "hidden_size": {"type": "int", "low": 32, "high": 128, "step": 8},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "num_attention_heads": {"type": "int", "low": 1, "high": 8},
            "lstm_layers": {"type": "int", "low": 1, "high": 3},
            "attn_dropout": {"type": "float", "low": 0.0, "high": 0.3},
            "add_relative_index": {"type": "categorical", "choices": [True, False]},
            "context_length_ratio": {"type": "float", "low": 0.5, "high": 1.0},
            "encoder_layers": {"type": "int", "low": 1, "high": 3},
        },
    }
