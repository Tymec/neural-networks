"""Layer"""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from nn.activation import Activation, ReLU
from nn.initializer import Initializer, get_bias_initializer, get_weight_initializer


class Layer(ABC):
    @abstractmethod
    def forward(self, inputs: npt.ArrayLike, training: bool = False) -> npt.ArrayLike: ...

    @abstractmethod
    def update_grads(self, dinputs: npt.ArrayLike) -> npt.ArrayLike: ...

    @abstractmethod
    def reset_grads(self) -> None: ...

    @abstractmethod
    def reset_cache(self) -> None: ...


class Linear(Layer):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        activation: Activation = ReLU(),
        init_weights: Initializer = None,
        init_biases: Initializer = None,
    ):
        self.activation = activation

        if init_weights is None:
            init_weights = get_weight_initializer(activation)
        if init_biases is None:
            init_biases = get_bias_initializer(activation)

        # Initialise weights and biases
        self.weights = init_weights((out_size, in_size))
        self.biases = init_biases((1, out_size))[0]

        # Cost gradients
        self.grad_weights = np.zeros(self.weights.shape)
        self.grad_biases = np.zeros(self.biases.shape)

        # Velocities
        self.vel_weights = np.zeros(self.weights.shape)
        self.vel_biases = np.zeros(self.biases.shape)

    def __repr__(self):
        in_size = self.weights.shape[1]
        out_size = self.weights.shape[0]
        return f"Layer(in={in_size}, out={out_size})"

    def forward(self, inputs: npt.ArrayLike, training: bool = False) -> npt.ArrayLike:
        # Calculate the weighted sum of inputs and add the bias
        # z = np.dot(self.weights, inputs) + self.biases
        z = self.weights @ inputs + self.biases  # TODO: Requires more testing

        # Store inputs and weighted inputs for backpropagation
        if training:
            self.inputs = inputs
            self.z = z

        # Apply the activation function
        return self.activation.f(z)

    def update_grads(self, dinputs: npt.ArrayLike) -> npt.ArrayLike:
        d_z = self.activation.df(self.z) * dinputs

        print(f"Z ({dinputs.shape}) = {dinputs}")
        print(f"X ({self.inputs.shape}) = {self.inputs}")
        print(f"dW ({self.activation.df(self.z).shape}) = {self.activation.df(self.z)}")
        print(f"dZ ({d_z.shape}) = {d_z}")
        print(f"grad_weights ({self.grad_weights.shape}) = {self.grad_weights}")

        # Calculate gradients
        self.grad_weights += np.outer(d_z, self.inputs)
        self.grad_biases += d_z

        # DEBUG
        # print(f"dA ({dinputs.shape}) = {dinputs}")
        # print(f"weights ({self.weights.shape}) = {self.weights}")
        # print(f"biases ({self.biases.shape}) = {self.biases}")
        # print(f"grad_biases ({self.grad_biases.shape}) = {self.grad_biases}")

        return d_z

    def reset_grads(self) -> None:
        # Reset gradients
        self.grad_weights = np.zeros(self.weights.shape)
        self.grad_biases = np.zeros(self.biases.shape)

    def reset_cache(self) -> None:
        # Reset inputs and weighted inputs
        self.inputs = None
        self.z = None
