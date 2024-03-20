"""Initializers"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Literal

import numpy as np
import numpy.typing as npt
from numpy.random import default_rng


def calculate_gain(
    nonlinearity: Literal["Linear", "ReLU", "LeakyRelu", "Tanh", "Sigmoid", "SELU"],
    param: float = 0.0,
) -> float:
    if nonlinearity == "Linear":
        return 1
    elif nonlinearity in ("ReLU", "LeakyRelu"):
        return np.sqrt(2 / (1 + param**2))
    elif nonlinearity == "Tanh":
        return 5 / 3
    elif nonlinearity == "Sigmoid":
        return 1
    elif nonlinearity == "SELU":
        return 3 / 4
    else:
        raise ValueError(f"Nonlinearity {nonlinearity!r} not supported.")


class FanMode(Enum):
    IN = "fan_in"
    OUT = "fan_out"


class Initializer(ABC):
    @abstractmethod
    def __call__(self, shape: tuple[int, ...]) -> npt.ArrayLike: ...


class BaseRandom(Initializer):
    def __init__(self, seed: int = 0):
        self.gen = default_rng(seed=seed)


class Constant(Initializer):
    def __init__(self, value: float = 0.0):
        self.value = value

    def __call__(self, shape: tuple[int, ...]) -> npt.ArrayLike:
        return np.full(shape, self.value)


class Zeros(Constant):
    def __call__(self, shape: tuple[int, ...]) -> npt.ArrayLike:
        return np.zeros(shape)


class Ones(Constant):
    def __call__(self, shape: tuple[int, ...]) -> npt.ArrayLike:
        return np.ones(shape)


class RandomUniform(BaseRandom):
    def __init__(self, minval: float = -1.0, maxval: float = 1.0, seed: int = 0):
        super().__init__(seed)
        self.minval = minval
        self.maxval = maxval

    def __call__(self, shape: tuple[int, ...]) -> npt.ArrayLike:
        return self.gen.uniform(self.minval, self.maxval, shape)


class RandomNormal(BaseRandom):
    def __init__(self, mean: float = 0.0, std: float = 1.0, seed: int = 0):
        super().__init__(seed)
        self.mean = mean
        self.std = std

    def __call__(self, shape: tuple[int, ...]) -> npt.ArrayLike:
        return self.gen.normal(self.mean, self.std, shape)


class XavierUniform(BaseRandom):
    def __init__(self, gain: float = 1.0, seed: int = 0):
        super().__init__(seed)
        self.gain = gain

    def __call__(self, shape: tuple[int, ...]) -> npt.ArrayLike:
        limit = self.gain * np.sqrt(6 / (shape[0] + shape[1]))
        return self.gen.uniform(-limit, limit, shape)


class XavierNormal(BaseRandom):
    def __init__(self, gain: float = 1.0, seed: int = 0):
        super().__init__(seed)
        self.gain = gain

    def __call__(self, shape: tuple[int, ...]) -> npt.ArrayLike:
        std = self.gain * np.sqrt(2 / (shape[0] + shape[1]))
        return self.gen.normal(0, std, shape)


class KaimingUniform(BaseRandom):
    def __init__(
        self,
        negative_slope: float = 0.0,
        mode: FanMode = FanMode.IN,
        nonlinearity: Literal["ReLU", "LeakyRelu"] = "LeakyRelu",
        seed: int = 0,
    ):
        super().__init__(seed)
        self.mode = mode
        self.gain = calculate_gain(nonlinearity, negative_slope)

    def __call__(self, shape: tuple[int, ...]) -> npt.ArrayLike:
        fan = shape[0] if self.mode == FanMode.IN else shape[1]
        limit = self.gain * np.sqrt(6 / fan)
        return self.gen.uniform(-limit, limit, shape)


class KaimingNormal(BaseRandom):
    def __init__(
        self,
        negative_slope: float = 0.0,
        mode: FanMode = FanMode.IN,
        nonlinearity: Literal["ReLU", "LeakyRelu"] = "LeakyRelu",
        seed: int = 0,
    ):
        super().__init__(seed)
        self.mode = mode
        self.gain = calculate_gain(nonlinearity, negative_slope)

    def __call__(self, shape: tuple[int, ...]) -> npt.ArrayLike:
        fan = shape[0] if self.mode == FanMode.IN else shape[1]
        std = self.gain * np.sqrt(2 / fan)
        return self.gen.normal(0, std, shape)


def get_weight_initializer(activation: str) -> Initializer:
    """Picks the appropriate initializer for the given activation function."""
    if activation in ("ReLU", "LeakyRelu"):
        return KaimingUniform(negative_slope=0.01, mode=FanMode.IN, nonlinearity=activation)
    elif activation in ("Tanh", "Sigmoid"):
        return XavierUniform(gain=calculate_gain(activation))
    elif activation == "SELU":
        return XavierNormal(gain=calculate_gain(activation))
    else:
        return XavierUniform()


def get_bias_initializer(activation: str) -> Initializer:
    """Picks the appropriate initializer for the given activation function."""
    if activation in ("ReLU", "LeakyRelu"):
        return Constant(value=0.01)
    else:
        return Zeros()


"""
MISSING:
Eye (2D)
Dirac (3D/4D/5D)
TruncatedNormal
Orthogonal
Sparse
"""
