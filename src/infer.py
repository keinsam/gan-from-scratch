import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import GAN
from dataset import get_dataloaders
import matplotlib.pyplot as plt

# Load hyperparameters
with open("configs/hparams.yaml", "r") as f:
    hparams = yaml.safe_load(f)

INPUT_DIM = hparams["model"]["input_dim"]
HIDDEN_DIM = hparams["model"]["hidden_dim"]
LATENT_DIM = hparams["model"]["latent_dim"]
BATCH_SIZE = hparams["train"]["batch_size"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load paths
with open("configs/paths.yaml", "r") as f:
    paths = yaml.safe_load(f)

MODEL_NAME = Path(paths["model_name"])
MODEL_DIR = Path(paths["model_dir"])
MODEL_PATH = MODEL_DIR.joinpath(f"{MODEL_NAME}.pth")

# Load dataloaders
train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

# TODO :
# Load the trained model
# Generate samples
# Plot samples