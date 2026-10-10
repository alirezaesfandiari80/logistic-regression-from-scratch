# Logistic Regression from Scratch

## Overview

This project explores Logistic Regression for binary classification, focusing on its mathematical foundations, parameter learning, and decision-boundary visualization.

## Theoretical Background

### 1. Linear Score

For an input feature vector `x`, the model computes a linear score:

$$
z = w^T x + b
$$

Here, `w` is the weight vector and `b` is the bias.

### 2. Sigmoid Function

The sigmoid function maps any real-valued score to a value between 0 and 1:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

The model estimates the probability of the positive class as:

$$
P(y=1 \mid x) = \sigma(w^T x + b)
$$

A threshold, commonly 0.5, can be used to convert the estimated probability into a class prediction.

### 3. Maximum Likelihood Estimation

For independent binary-labeled examples, define `p_i = sigmoid(w^T x_i + b)`. The likelihood is:

$$
L(w,b) = \prod_{i=1}^{n} p_i^{y_i}(1-p_i)^{1-y_i}
$$

Maximizing the likelihood is equivalent to minimizing the negative log-likelihood, known as binary cross-entropy:

$$
J(w,b) = -\frac{1}{n}\sum_{i=1}^{n}
\left[y_i\log(p_i) + (1-y_i)\log(1-p_i)\right]
$$

### 4. Gradient Descent

For the average binary cross-entropy objective, the gradients are:

$$
\nabla_w J = \frac{1}{n}X^T(p-y)
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{n}\sum_{i=1}^{n}(p_i-y_i)
$$

Gradient descent updates the parameters to reduce the loss:

$$
w \leftarrow w-\eta\nabla_w J
$$

$$
b \leftarrow b-\eta\frac{\partial J}{\partial b}
$$

Here, `η` is the learning rate, `X` is the feature matrix, and `p` contains the predicted probabilities.

## What This Project Explores

- Logistic Regression and the sigmoid function
- Learning weights and bias from data
- Loss minimization and gradient-based optimization
- Classification performance and decision-boundary visualization

## How to Run

1. Open the notebook in Jupyter Notebook or Google Colab.
2. Install the libraries imported by the notebook.
3. Run the cells in order.

## Learning Objective

The goal is to connect the mathematical formulation of Logistic Regression with its implementation and observed behavior during training.
