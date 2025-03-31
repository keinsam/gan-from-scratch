# Generative Adversarial Networks (GAN) in PyTorch

## Overview

This repository contains an implementation of **Generative Adversarial Networks (GAN)** using PyTorch, based on the original paper by **Goodfellow et al. (2014)**.

## Features

- Implementation of a **GAN** in PyTorch.
- Training on the **MNIST dataset**.
- Logging with TensorBoard.
- Generation of images with visualization in Plotly.

## Usage

### Training
To train the GAN, run:
```bash
python train.py
```
This script will:
- Train the model.
- Save the trained model in `MODEL_DIR`.
- Log the loss values in TensorBoard.

To see the plotted loss functions, run ```tensorboard --logdir=logs``` in the terminal.

### Inference
To visualize generated images from the trained model, run:
```bash
python infer.py
```
This script will:
- Load the trained model.
- Display generated images.

## References

- **Generative Adversarial Networks** – Goodfellow et al. (2014) [[Paper](https://arxiv.org/abs/1406.2661)]