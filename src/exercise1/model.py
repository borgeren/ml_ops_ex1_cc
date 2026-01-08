import torch
from torch import nn
import hydra
from omegaconf import DictConfig

import logging

log = logging.getLogger(__name__)

class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self, dropout_rate: float = 0.5) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)

@hydra.main(version_base="1.3", config_path="./configs", config_name="model_conf.yaml")
def main(cfg: DictConfig) -> None:
    model = MyAwesomeModel()
    log.info(f"Model architecture: {model}")
    log.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    log.info(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()
