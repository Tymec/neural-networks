from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


def plot_function(
    f: Callable[[float], float],
    x_min: float = -10,
    x_max: float = 10,
    y_min: float = -1.2,
    y_max: float = 1.2,
    ax: plt.Axes = None,
    label: str = None,
) -> plt.Axes:
    x_step = (x_max - x_min) / 100

    xs = np.arange(x_min, x_max, x_step)
    ys = [f(x) for x in xs]

    if ax is None:
        ax = plt.gca()

    ax.set_xlim([x_min - 1, x_max + 1])
    ax.set_ylim([y_min, y_max])

    ax.plot(xs, ys, color="b", linewidth=1.5)
    ax.locator_params(nbins=4)
    ax.grid(True)

    if label is not None:
        ax.set_xlabel(label)

    return ax


def decision_boundary(X: list, y: list, fn: Callable) -> tuple[list, list, list]:
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = fn([X[i, j], Y[i, j]])

    return X, Y, Z
