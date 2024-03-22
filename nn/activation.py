"""Activation functions"""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

# TODO: Learnable parameters


class Activation(ABC):
    def __call__(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return self.f(x)

    @abstractmethod
    def f(self, x: npt.ArrayLike) -> npt.ArrayLike: ...

    @abstractmethod
    def df(self, x: npt.ArrayLike) -> npt.ArrayLike: ...


class Identity(Activation):
    """Linear activation function"""

    def f(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return x

    def df(self, _: npt.ArrayLike) -> npt.ArrayLike:
        return 1


class BinaryStep(Activation):
    """Binary step activation function"""

    def f(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return np.heaviside(x, 1)

    def df(self, _: npt.ArrayLike) -> npt.ArrayLike:
        return 0


class Sigmoid(Activation):
    """Sigmoid activation function"""

    def f(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return 1 / (1 + np.exp(-x))

    def df(self, x: npt.ArrayLike) -> npt.ArrayLike:
        y = self.f(x)
        return y * (1 - y)


class LogSigmoid(Sigmoid):
    """Logarithm of the sigmoid activation function"""

    def f(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return np.log(super().f(x))

    def df(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return super().f(-x)


class Tanh(Activation):
    """Hyperbolic tangent activation function"""

    def f(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return np.tanh(x)

    def df(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return 1 - np.tanh(x) ** 2


class LeakyReLU(Activation):
    """Leaky rectified linear unit activation function"""

    def __init__(self, negative_slope: float = 0.01):
        assert negative_slope >= 0, "negative_slope must be positive"
        self.negative_slope = negative_slope

    def f(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return np.maximum(0, x) + self.negative_slope * np.minimum(0, x)

    def df(self, x: npt.ArrayLike) -> npt.ArrayLike:
        # ReLU'(0) = 0 (source: https://openreview.net/forum?id=urrcVI-_jRm)
        return 1 * (x > 0) + self.negative_slope * (x <= 0)


class ReLU(LeakyReLU):
    """Rectified linear unit activation function"""

    def __init__(self):
        super().__init__(0)


class Swish(Sigmoid):
    """Signmoid-weighted linear activation function"""

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def f(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return x * super().f(self.beta * x)

    def df(self, x: npt.ArrayLike) -> npt.ArrayLike:
        y = super().f(self.beta * x)
        return y + self.f(x) * (1 - y)


class SiLU(Swish):
    """Sigmoid linear unit activation function"""

    def __init__(self):
        super().__init__(1)


class ELU(Activation):
    """Exponential linear unit activation function"""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def f(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def df(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return np.where(x > 0, 1, self.f(x) + self.alpha)


"""
Hardshrink
Hardsigmoid
Hardtanh
Hardswish
MultiheadAttention
PReLU (trainable)
ReLU6
RReLU
    SELU
CELU
GELU
Mish
Softplus
Softshrink
Softsign
Tanhshrink
Threshold
GLU
"""


class Softmax(Activation):
    """Softmax activation function"""

    def f(self, x: npt.ArrayLike) -> npt.ArrayLike:
        z = np.exp(x - np.max(x))  # to help with numerical stability
        return z / np.sum(z, axis=0, keepdims=True)  # TODO: axis

    def df(self, x: npt.ArrayLike) -> npt.ArrayLike:
        z = np.reshape(x, (-1, 1))
        return np.diagflat(z) - np.dot(z, z.T)


"""
Softmin
Softmax
Softmax2d
LogSoftmax
AdaptiveLogSoftmaxWithLoss
"""
