# Activation functions
import math


def identity(x: float) -> float:
    return x


def binary_step(x: float) -> float:
    return 1 if x >= 0 else 0


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def relu(x: float) -> float:
    return max(0, x)


def tanh(x: float) -> float:
    return math.tanh(x)


def selu(x: float) -> float:
    alpha = 1.67326324
    scale = 1.05070098
    return scale * (x if x >= 0 else alpha * (math.exp(x) - 1))
