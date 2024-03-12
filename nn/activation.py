"""Activation functions"""

from abc import ABC

import numpy as np
import numpy.typing as npt


class Activation0D(ABC):
    def __init__(self):
        pass

    def f(self, x: npt.ArrayLike) -> npt.ArrayLike:
        raise NotImplementedError

    def df(self, x: npt.ArrayLike) -> npt.ArrayLike:
        raise NotImplementedError


class Activation1D(Activation0D):
    def __init__(self, alpha: float):
        self.alpha = alpha


class Activation2D(Activation1D):
    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta


class ELU(Activation1D):
    def __init__(self, alpha: float = 1.0):
        super().__init__(alpha)

    def f(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return x * (x > 0) + self.alpha * (np.exp(x) - 1) * (x <= 0)

    def df(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return (x > 0) + self.alpha * np.exp(x) * (x <= 0)


class Hardtanh(Activation2D):
    def __init__(self, alpha: float = -1.0, beta: float = 1.0):
        super().__init__(alpha, beta)

    def f(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return np.clip(x, self.alpha, self.beta)

    def df(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return (self.alpha <= x) * (x <= self.beta)


class Hardshrink(Hardtanh):
    def __init__(self, lambd: float = 0.5):
        super().__init__(lambd, -lambd)

    def f(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return np.clip(x, self.alpha, self.beta)

    def df(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return (self.alpha <= x) * (x <= self.beta)


class Hardsigmoid(Activation0D):
    def f(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return np.clip(x / 6 + 1 / 2, 0, 1)

    def df(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return (0 <= x) * (x <= 1) / 6


class ReLU(Activation0D):
    def f(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return x * (x > 0)

    def df(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return x > 0


class Sigmoid(Activation0D):
    def f(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return 1 / (1 + np.exp(-x))

    def df(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return x * (1 - x)


class Tanh(Activation0D):
    def f(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return np.tanh(x)

    def df(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return 1 - x**2


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
    Sigmoid
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
