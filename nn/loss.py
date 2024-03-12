"""Loss functions"""

import numpy as np
import numpy.typing as npt


class MeanSquaredError:
    @staticmethod
    def f(y_pred: npt.ArrayLike, y_true: npt.ArrayLike) -> npt.ArrayLike:
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def df(y_pred: npt.ArrayLike, y_true: npt.ArrayLike) -> npt.ArrayLike:
        return y_pred - y_true
