# test_base_classes.py
import torch
import torch.nn as nn
from base_config import BaseConfig
from base_lit_model import BaseLitModel


# Create a test config class
class TestConfig(BaseConfig):
    MODEL_PARAMS = ["test_param"]

    def __init__(
        self,
        input_len: int,
        output_len: int,
        input_size: int,
        static_size: int = 0,
        test_param: int = 42,
        **kwargs,
    ):
        super().__init__(
            input_len=input_len,
            output_len=output_len,
            input_size=input_size,
            static_size=static_size,
            test_param=test_param,
            **kwargs,
        )


# Create a minimal model class
class TestModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.input_size, config.output_len)

    def forward(self, x, static=None, future=None):
        return self.linear(x[:, -1]).unsqueeze(-1)


# Create a test lightning module
class TestLitModel(BaseLitModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = TestModel(config)

    def forward(self, x, static=None, future=None):
        return self.model(x, static, future)


# Test the classes
config = TestConfig(input_len=10, output_len=5, input_size=8, test_param=100)
print(f"Config: {config.to_dict()}")

model = TestLitModel(config)
print(f"Model hyperparameters: {model.hparams}")

# Test forward pass with dummy data
x = torch.randn(2, 10, 8)  # [batch_size, input_len, input_size]
static = torch.randn(2, 5)  # [batch_size, static_size]
output = model(x, static)
print(f"Output shape: {output.shape}")  # Should be [2, 5, 1]

print("Base classes test successful!")
