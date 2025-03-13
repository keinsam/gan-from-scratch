import yaml
from pathlib import Path
import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from model import GAN

# Load hyperparameters
with open("configs/hparams.yaml", "r") as f:
    hparams = yaml.safe_load(f)

INPUT_DIM = hparams["model"]["input_dim"]
HIDDEN_DIM = hparams["model"]["hidden_dim"]
LATENT_DIM = hparams["model"]["latent_dim"]
BATCH_SIZE = hparams["train"]["batch_size"]
NB_EPOCHS = hparams["train"]["nb_epochs"]
LEARNING_RATE = hparams["train"]["learning_rate"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load paths
with open("configs/paths.yaml", "r") as f:
    paths = yaml.safe_load(f)

MODEL_NAME = Path(paths["model_name"])
LOG_DIR = Path(paths["log_dir"])
MODEL_DIR = Path(paths["model_dir"])
MODEL_PATH = MODEL_DIR.joinpath(f"{MODEL_NAME}.pth")
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=LOG_DIR.joinpath(MODEL_NAME))
writer.add_hparams({"input_dim": INPUT_DIM, "hidden_dim": HIDDEN_DIM, "latent_dim": LATENT_DIM,
                    "batch_size": BATCH_SIZE, "nb_epochs": NB_EPOCHS, "learning_rate": LEARNING_RATE,},
                    {})

# TODO :
# Load dataloaders
# Initialize model and optimizer
# Training loop
# Save model
# Close TensorBoard writer