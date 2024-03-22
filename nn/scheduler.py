"""LR Schedulers"""

from abc import ABC, abstractmethod


class Scheduler(ABC):
    def __call__(self, lr: float) -> float:
        return self.get_lr(lr)

    @abstractmethod
    def get_lr(self, lr: float) -> float: ...


class StaticLR(Scheduler):
    def get_lr(self, lr: float) -> float:
        return lr


class ConstantLR(Scheduler):
    def __init__(self, factor: float = 1 / 3, epochs: int = 5):
        self.factor = factor
        self.epochs = epochs

    def get_lr(self, lr: float) -> float:
        if self.epochs <= 0:
            return lr

        self.epochs -= 1
        return lr * self.factor


"""
    ConstantLR
LambdaLR
MultiplicativeLR
StepLR
MultiStepLR
LinearLR
ExponentialLR
PolynomialLR
CosineAnnealingLR
ChainedScheduler
SequentialLR
ReduceLROnPlateau
CyclicLR
OneCycleLR
CosineAnnealingWarmRestarts
"""
