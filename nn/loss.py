"""Loss functions"""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class Loss(ABC):
    def __init__(self, mean: bool = True):
        self.mean = mean

    def __call__(self, y_pred: npt.ArrayLike, y_true: npt.ArrayLike) -> float:
        cost = self.f(y_pred, y_true)
        return np.mean(cost) if self.mean else np.sum(cost)

    @abstractmethod
    def f(self, y_pred: npt.ArrayLike, y_true: npt.ArrayLike) -> npt.ArrayLike: ...

    @abstractmethod
    def df(self, y_pred: npt.ArrayLike, y_true: npt.ArrayLike) -> npt.ArrayLike: ...


class MeanSquaredError(Loss):
    def f(self, y_pred: npt.ArrayLike, y_true: npt.ArrayLike) -> npt.ArrayLike:
        return (y_true - y_pred) ** 2

    def df(self, y_pred: npt.ArrayLike, y_true: npt.ArrayLike) -> npt.ArrayLike:
        return 2 * (y_pred - y_true)


"""
    MSELoss
L1Loss
CrossEntropyLoss
CTCLoss
NLLLoss
PoissonNLLLoss
GaussianNLLLoss
KLDivLoss
BCELoss
BCEWithLogitsLoss
MarginRankingLoss
HingeEmbeddingLoss
MultiLabelMarginLoss
HuberLoss
SmoothL1Loss
SoftMarginLoss
MultiLabelSoftMarginLoss
CosineEmbeddingLoss
MultiMarginLoss
TripletMarginLoss
TripletMarginWithDistanceLoss
"""
