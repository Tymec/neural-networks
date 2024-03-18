"""Activation functions"""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class Activation(ABC):
    @abstractmethod
    def f(self, x: npt.ArrayLike) -> npt.ArrayLike: ...

    @abstractmethod
    def df(self, x: npt.ArrayLike) -> npt.ArrayLike: ...


class Activation1D(Activation):
    def __init__(self, alpha: float):
        self.alpha = alpha


class Activation2D(Activation):
    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta


class Identity(Activation):
    def f(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return x

    def df(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return 1


class BinaryStep(Activation):
    def f(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return np.heaviside(x, 1)

    def df(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return 0


class Sigmoid(Activation):
    def f(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return 1 / (1 + np.exp(-x))

    def df(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return x * (1 - x)


class ReLU(Activation):
    def f(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return np.maximum(0.0, x)

    def df(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return (x > 0).astype(int)


"""
ELU
Hardshrink
Hardsigmoid
Hardtanh
Hardswish
LeakyReLU
LogSigmoid
MultiheadAttention
PReLU (trainable)
ReLU
ReLU6
RReLU
SELU
CELU
GELU
SiLU
Mish
Softplus
Softshrink
Softsign
Tanh
Tanhshrink
Threshold
GLU

Softmin
Softmax
Softmax2d
LogSoftmax
AdaptiveLogSoftmaxWithLoss
"""
