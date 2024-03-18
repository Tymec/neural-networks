"""Optimizers"""

from abc import ABC, abstractmethod

from nn.layer import Layer


class Optimizer(ABC):
    @property
    def lr(self) -> float:
        return self._lr

    @lr.setter
    def lr(self, learning_rate: float) -> None:
        self._lr = learning_rate

    @abstractmethod
    def apply(self, layer: Layer) -> None: ...


class GradientDescent(Optimizer):
    def __init__(self, learning_rate: float = 0.001, momentum: float = 0.0):
        # TODO: Weight decay
        # TODO: Dampening (for momentum)
        # TODO: Nesterov (alternative momentum algorithm)
        # TODO: Maximize
        self.lr = learning_rate
        self.momentum = momentum

    def apply(self, layer: Layer) -> None:
        # Update velocities
        layer.vel_weights = self.momentum * layer.vel_weights + self.lr * layer.grad_weights
        layer.vel_biases = self.momentum * layer.vel_biases + self.lr * layer.grad_biases

        # Update weights and biases
        layer.weights -= layer.vel_weights
        layer.biases -= layer.vel_biases


"""
Adadelta
Adagrad
Adam
AdamW
SparseAdam
Adamax
ASGD
LBFGS
NAdam
RAdam
RMSprop
Rprop
"""
