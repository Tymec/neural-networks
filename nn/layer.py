"""Layer"""

import numpy as np
import numpy.typing as npt

from nn.activation import ReLU
from nn.initializer import get_bias_initializer, get_weight_initializer


class Layer:
    def __init__(
        self,
        in_size: int,
        out_size: int,
        activation=ReLU(),
        init_weights=None,
        init_biases=None,
    ):
        self.activation = activation

        if init_weights is None:
            init_weights = get_weight_initializer(activation.__name__)
        if init_biases is None:
            init_biases = get_bias_initializer(activation.__name__)

        # Initialise weights and biases to random values
        # self.weights = np.random.rand(out_size, in_size)
        # self.biases = np.random.rand(1, out_size)[0]
        self.weights = init_weights.f(out_size, in_size)
        self.biases = init_biases.f(1, out_size)[0]

        self.grad_weights = np.zeros(self.weights.shape)
        self.grad_biases = np.zeros(self.biases.shape)

    def __repr__(self):
        in_size = self.weights.shape[1]
        out_size = self.weights.shape[0]
        return f"Layer(in={in_size}, out={out_size}, activation={self.activation.__name__})"

    def forward(self, inputs: npt.ArrayLike, training: bool = False) -> npt.ArrayLike:
        # Calculate the weighted sum of inputs and add the bias
        z = np.dot(self.weights, inputs) + self.biases

        # Store inputs and weighted inputs for backpropagation
        if training:
            self.inputs = inputs
            self.z = z

        # Apply the activation function
        return self.activation.f(z)

    def update_grads(self, dinputs: npt.ArrayLike) -> npt.ArrayLike:
        # Calculate the gradient of the loss with respect to the weighted sum
        d_z = dinputs * self.activation.df(self.z)
        self.grad_weights += np.dot(d_z[:, None], self.inputs[None, :])
        self.grad_biases += d_z
        return d_z

    def apply_grads(self, lr: float) -> None:
        # Update weights and biases
        self.weights -= lr * self.grad_weights
        self.biases -= lr * self.grad_biases

        # Reset gradients
        self.grad_weights = np.zeros(self.weights.shape)
        self.grad_biases = np.zeros(self.biases.shape)

        # Reset inputs and weighted inputs
        self.inputs = None
        self.z = None
