"""Loss functions"""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class Loss(ABC):
    @staticmethod
    @abstractmethod
    def f(y_pred: npt.ArrayLike, y_true: npt.ArrayLike) -> npt.ArrayLike: ...

    @staticmethod
    @abstractmethod
    def df(y_pred: npt.ArrayLike, y_true: npt.ArrayLike) -> npt.ArrayLike: ...


class MeanSquaredError:
    @staticmethod
    def f(y_pred: npt.ArrayLike, y_true: npt.ArrayLike) -> npt.ArrayLike:
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def df(y_pred: npt.ArrayLike, y_true: npt.ArrayLike) -> npt.ArrayLike:
        return y_pred - y_true


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
