# Generative Adversarial Networks (GAN) in PyTorch

## Overview

This repository contains an implementation of **Generative Adversarial Networks (GAN)** using PyTorch, based on the original paper by **Goodfellow et al. (2014)**, with improvements from later works such as DCGAN and various training stabilization techniques.

## Features

- Implementation of a **GAN** in PyTorch.
- Training on the **MNIST and CIFAR10** datasets.
- Logging with TensorBoard.

## Usage

To train the GAN, run:
```bash
python run.py
```
This script will:
- Train the model.
- Save the trained model in `MODEL_DIR`.
- Log the loss values and generated samples in TensorBoard.

To see the plotted loss functions and generated samples, run ```tensorboard --logdir=logs``` in the terminal.

## References

- **Generative Adversarial Networks** – Goodfellow et al. (2014) [[Paper](https://arxiv.org/abs/1406.2661)]
- **Unsupervised Representation Learning with Deep Convolutional GANs (DCGAN)** – Radford et al. (2015) [[Paper](https://arxiv.org/abs/1511.06434)]
- **Improved Techniques for Training GANs** – Salimans et al. (2016) [[Paper](https://arxiv.org/abs/1606.03498)]
