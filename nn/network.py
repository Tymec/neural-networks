"""Network"""

import numpy as np
import numpy.typing as npt

from nn.layer import Layer
from nn.loss import Loss, MeanSquaredError
from nn.optimizer import GradientDescent, Optimizer
from nn.scheduler import Scheduler, StaticLR

### TODO: Serialization
### TODO: Regularization


class Network:
    def __init__(
        self,
        layers: list[Layer],
        loss: Loss = MeanSquaredError,
        optim: Optimizer = GradientDescent(),
        scheduler: Scheduler = StaticLR(),
    ):
        self.epoch = 0

        self.layers = layers
        self.loss = loss
        self.optimizer = optim
        self.scheduler = scheduler

    def __repr__(self):
        return f"Network(layers={self.layers}, loss={self.loss.__name__}, optim={self.optimizer.__name__}, scheduler={self.scheduler.__name__})"

    @staticmethod
    def _batch(
        X: npt.ArrayLike,
        Y: npt.ArrayLike,
        batch_size: int,
        even: bool = False,
    ) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        if even:
            np.array_split(X, len(X) / batch_size)
            np.array_split(Y, len(Y) / batch_size)
            return X, Y  # Unnecessary since above functions split in place

        # Random
        indexes = np.random.randint(len(X), size=batch_size)
        return X[indexes, :], Y[indexes, :]

    def update_grads(self, inputs: npt.ArrayLike, targets: npt.ArrayLike) -> float:
        # Forward pass
        predictions = self.forward(inputs, training=True)

        # Backward pass
        dinputs = self.loss.df(predictions, targets)
        for layer in reversed(self.layers):
            d_z = layer.update_grads(dinputs)
            dinputs = np.dot(layer.weights.T, d_z)

        return self.loss.f(predictions, targets)

    def forward(self, inputs: npt.ArrayLike, training: bool = False) -> npt.ArrayLike:
        for layer in self.layers:
            inputs = layer.forward(inputs, training)

        return inputs

    def train(self, input_data: npt.ArrayLike, target_data: npt.ArrayLike, batch_size: int = -1) -> float:
        """
        Batch gradient descent: batch_size == len(input_data)
        Mini-batch gradient descent: 1 < batch_size < len(input_data)
        Stochastic gradient descent: batch_size == 1
        """
        if batch_size <= 0 or batch_size >= len(input_data):
            # Train on the entire dataset
            batch_size = len(input_data)

        # Split data into batches
        X, Y = self._batch(input_data, target_data, batch_size=batch_size)

        iters = 0  # TODO: do something with this
        loss = 0
        for x, y in zip(X, Y):
            loss += self.train_batch(x, y)
            iters += batch_size

        # Learning rate scheduling
        self.optimizer.lr = self.scheduler.get_lr(self.optimizer.lr)  # TODO: Make this nicer

        self.epoch += 1
        return loss / len(X)

    def train_batch(self, input_data: npt.ArrayLike, target_data: npt.ArrayLike) -> float:
        loss = 0
        for inputs, targets in zip(input_data, target_data):
            loss += self.update_grads(inputs, targets)

        for layer in self.layers:
            self.optimizer.apply(layer)
            layer.reset_grads()
            layer.reset_cache()

        return loss / len(input_data)
