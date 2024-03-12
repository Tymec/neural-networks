"""Network"""


class Network:
    def __init__(self, layers: list[Layer], lr: float = 0.01, loss=MeanSquaredError):
        self.layers = layers
        self.learning_rate = lr
        self.loss = loss

    def __repr__(self):
        return f"Network({self.layers}, lr={self.learning_rate}, loss={self.loss.__name__})"

    def forward(self, inputs: npt.ArrayLike, training: bool = False) -> npt.ArrayLike:
        for layer in self.layers:
            inputs = layer.forward(inputs, training)

        return inputs

    def train(self, input_data: npt.ArrayLike, target_data: npt.ArrayLike) -> float:
        loss = 0
        for inputs, targets in zip(input_data, target_data):
            loss += self.update_grads(inputs, targets)

        for layer in self.layers:
            layer.apply_grads(self.learning_rate)

        return loss / len(input_data)

    def update_grads(self, inputs: npt.ArrayLike, targets: npt.ArrayLike) -> float:
        # Forward pass
        predictions = self.forward(inputs, training=True)

        # Backward pass
        dinputs = self.loss.df(predictions, targets)
        for layer in reversed(self.layers):
            d_z = layer.update_grads(dinputs)
            dinputs = np.dot(layer.weights.T, d_z)

        return self.loss.f(predictions, targets)
