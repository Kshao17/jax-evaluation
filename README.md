# Pet Pawpularity Prediction with JAX

This project predicts pet photo Pawpularity scores using computer vision and regression.

The main focus of the project is refactoring a PyTorch-based image regression pipeline into JAX/Flax while explicitly managing model state, gradients, optimization, and validation logic.

## Project Overview

The task is to predict a Pawpularity score from pet images.

The original baseline used a PyTorch pipeline with a pretrained ResNet-18 backbone and a custom MLP regression head.

This project refactors that workflow into JAX/Flax to explore functional programming, explicit model-state management, and custom training loops.

## Main Goals

- Refactor a PyTorch computer vision model into JAX/Flax
- Implement a functional regression model
- Explicitly manage BatchNorm state
- Implement RMSE loss in JAX
- Build custom training and validation steps
- Compare JAX and PyTorch development patterns
- Analyze model generalization and overfitting behavior

## Model Architecture

The model uses a pretrained ResNet-18 backbone to extract image features.

The JAX/Flax version:

1. Extracts feature maps from the ResNet-18 backbone
2. Applies global average pooling using `jnp.mean`
3. Passes the pooled feature vector through a custom MLP head
4. Outputs a Pawpularity score for regression

The MLP head is implemented using `flax.linen` with dense layers and ReLU activations.

## PyTorch to JAX Refactoring

A major part of this project was converting the original object-oriented PyTorch workflow into JAX/Flax.

Key differences explored include:

### PyTorch

- Object-oriented model definition
- Implicit state management
- Simple `model.train()` / `model.eval()` workflow
- Easier debugging and rapid experimentation

### JAX / Flax

- Functional model design
- Explicit state management
- Pure loss functions
- Explicit gradient computation
- JIT-compatible training functions
- Greater control over reproducibility and transformations

## Training Pipeline

The JAX training pipeline explicitly manages:

- Model parameters
- Batch normalization statistics
- Optimizer state
- Forward pass
- Loss computation
- Gradient computation
- Parameter updates

Training state is managed using `flax.training.TrainState`.

The optimizer used is AdamW through Optax.

## Loss Function

The project uses Root Mean Squared Error (RMSE):

```text
RMSE = sqrt(mean((prediction - target)^2))
