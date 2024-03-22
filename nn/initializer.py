"""Initializers"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Literal

import numpy as np
import numpy.typing as npt
from numpy.random import default_rng

import nn.activation as act


def calculate_gain(activation: act.Activation, param: float = 0.0) -> float:
    match type(activation):
        case act.ReLU | act.LeakyReLU:
            return np.sqrt(2 / (1 + param**2))
        case act.Tanh:
            return 5 / 3
        case act.SELU:
            return 3 / 4
        case _:
            return 1


class FanMode(Enum):
    IN = "fan_in"
    OUT = "fan_out"


class Initializer(ABC):
    def __call__(self, shape: tuple[int, ...]) -> npt.ArrayLike:
        return self.initialize(shape)

    @abstractmethod
    def initialize(self, shape: tuple[int, ...]) -> npt.ArrayLike: ...


class BaseRandom(Initializer):
    def __init__(self, seed: int = 0):
        self.gen = default_rng(seed=seed)


class Constant(Initializer):
    def __init__(self, value: float = 0.0):
        self.value = value

    def initialize(self, shape: tuple[int, ...]) -> npt.ArrayLike:
        return np.full(shape, self.value)


class Zeros(Constant):
    def initialize(self, shape: tuple[int, ...]) -> npt.ArrayLike:
        return np.zeros(shape)


class Ones(Constant):
    def initialize(self, shape: tuple[int, ...]) -> npt.ArrayLike:
        return np.ones(shape)


class RandomUniform(BaseRandom):
    def __init__(self, minval: float = -1.0, maxval: float = 1.0, seed: int = 0):
        super().__init__(seed)
        self.minval = minval
        self.maxval = maxval

    def initialize(self, shape: tuple[int, ...]) -> npt.ArrayLike:
        return self.gen.uniform(self.minval, self.maxval, shape)


class RandomNormal(BaseRandom):
    def __init__(self, mean: float = 0.0, std: float = 1.0, seed: int = 0):
        super().__init__(seed)
        self.mean = mean
        self.std = std

    def initialize(self, shape: tuple[int, ...]) -> npt.ArrayLike:
        return self.gen.normal(self.mean, self.std, shape)


class XavierUniform(BaseRandom):
    def __init__(self, gain: float = 1.0, seed: int = 0):
        super().__init__(seed)
        self.gain = gain

    def initialize(self, shape: tuple[int, ...]) -> npt.ArrayLike:
        limit = self.gain * np.sqrt(6 / (shape[0] + shape[1]))
        return self.gen.uniform(-limit, limit, shape)


class XavierNormal(BaseRandom):
    def __init__(self, gain: float = 1.0, seed: int = 0):
        super().__init__(seed)
        self.gain = gain

    def initialize(self, shape: tuple[int, ...]) -> npt.ArrayLike:
        std = self.gain * np.sqrt(2 / (shape[0] + shape[1]))
        return self.gen.normal(0, std, shape)


class KaimingUniform(BaseRandom):
    def __init__(
        self,
        negative_slope: float = 0.0,
        mode: FanMode = FanMode.IN,
        nonlinearity: Literal["ReLU", "LeakyReLU"] = "LeakyReLU",
        seed: int = 0,
    ):
        super().__init__(seed)
        self.mode = mode
        self.gain = calculate_gain(nonlinearity, negative_slope)

    def initialize(self, shape: tuple[int, ...]) -> npt.ArrayLike:
        fan = shape[0] if self.mode == FanMode.IN else shape[1]
        limit = self.gain * np.sqrt(6 / fan)
        return self.gen.uniform(-limit, limit, shape)


class KaimingNormal(BaseRandom):
    def __init__(
        self,
        negative_slope: float = 0.0,
        mode: FanMode = FanMode.IN,
        nonlinearity: Literal["ReLU", "LeakyReLU"] = "LeakyReLU",
        seed: int = 0,
    ):
        super().__init__(seed)
        self.mode = mode
        self.gain = calculate_gain(nonlinearity, negative_slope)

    def initialize(self, shape: tuple[int, ...]) -> npt.ArrayLike:
        fan = shape[0] if self.mode == FanMode.IN else shape[1]
        std = self.gain * np.sqrt(2 / fan)
        return self.gen.normal(0, std, shape)


def get_weight_initializer(activation: str) -> Initializer:
    """Picks the appropriate initializer for the given activation function."""
    match activation:
        case "ReLU" | "LeakyReLU":
            return KaimingUniform(negative_slope=0.01, mode=FanMode.IN, nonlinearity=activation)
        case "Tanh" | "Sigmoid":
            return XavierUniform(gain=calculate_gain(activation))
        case "SELU":
            return XavierNormal(gain=calculate_gain(activation))
        case _:
            return XavierUniform()


def get_bias_initializer(activation: str) -> Initializer:
    """Picks the appropriate initializer for the given activation function."""
    match activation:
        case "ReLU" | "LeakyRelu":
            return Constant(value=0.01)
        case _:
            return Zeros()


"""
MISSING:
Eye (2D)
Dirac (3D/4D/5D)
TruncatedNormal
Orthogonal
Sparse
"""
