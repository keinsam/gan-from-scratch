# Generative Adversarial Network (GAN) in PyTorch

## Overview

This repository contains an implementation of a **Generative Adversarial Network (GAN)** using PyTorch, based on the original paper by **Goodfellow et al. (2014)**.

## Features

- Implementation of a **fully connected GAN** in PyTorch.
- Training on the **MNIST dataset**.
- Logging with **TensorBoard**.
- Generation of synthetic images for visualization.

## Project Structure
```
├── configs/
│   ├── hparams.yaml       # Hyperparameters configuration
│   ├── paths.yaml         # Paths for saving models/logs
│
├── src/
    ├── dataset.py         # Data loading utilities
    ├── infer.py           # Inference script
    ├── model.py           # GAN model (Generator & Discriminator)
    ├── train.py           # Training script
```

## Model Architecture

The GAN consists of:
- A **Generator** that maps random noise to generated images.
- A **Discriminator** that classifies images as real or fake.
- An **adversarial training setup**, where the Generator and Discriminator are trained in competition.

## Usage

### Training
To train the GAN, run:
```bash
python train.py
```
This script will:
- Train the model for the specified number of epochs.
- Save the trained model in `MODEL_DIR`.
- Log the loss values in TensorBoard.

### Inference
To generate and visualize synthetic images from the trained model, run:
```bash
python infer.py
```
This script will:
- Load the trained model.
- Generate and display synthetic images from random noise.

## References

- **Generative Adversarial Networks** – Goodfellow et al. (2014) [[Paper](https://arxiv.org/abs/1406.2661)]