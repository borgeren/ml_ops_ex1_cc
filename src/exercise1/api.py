import matplotlib.pyplot as plt
import torch
import typer
from data import corrupt_mnist
from model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


if __name__ == "__main__":
    print("Hello World from the API module of exercise 1!")