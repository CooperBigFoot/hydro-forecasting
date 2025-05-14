from typing import Any


def get_search_space() -> dict[str, dict[str, Any]]:
    """
    Define the hyperparameter search space for TiDE models.

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
            "num_encoder_layers": {"type": "int", "low": 1, "high": 3},
            "num_decoder_layers": {"type": "int", "low": 1, "high": 3},
            "decoder_output_size": {"type": "int", "low": 8, "high": 32},
            "temporal_decoder_hidden_size": {"type": "int", "low": 16, "high": 64},
            "use_layer_norm": {"type": "categorical", "choices": [True, False]},
        },
    }
