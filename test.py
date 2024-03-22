import numpy as np

import nn.activation as act
from nn.layer import Linear
from nn.loss import MeanSquaredError
from nn.network import Network

loss = MeanSquaredError()

y_pred = np.array([0.1, 0.2, 0.3])
y_true = np.array([0.1, 0.2, 0.3])

print("Loss should be 0.0 and is " + str(loss.f(y_pred, y_true)))
print("Gradient should be [0.0, 0.0, 0.0] and is " + str(loss.df(y_pred, y_true)))
print("")

y_pred = np.array([0.1, 0.2, 0.3])
y_true = np.array([0.2, 0.2, 0.3])

print("Loss should be 0.01 and is " + str(loss.f(y_pred, y_true)))
print("Gradient should be [-0.2, 0.0, 0.0] and is " + str(loss.df(y_pred, y_true)))

# Test ReLU'(0)
print("ReLU'(0) should be 0.0 and is " + str(act.ReLU().df(0)))


# Softmax test
results = []
# X = np.array([[1, 2, 3, 6], [2, 4, 5, 6], [1, 2, 3, 6]])
# Y = np.array(
#     [
#         [0.00626879, 0.01704033, 0.04632042, 0.93037045],
#         [0.01203764, 0.08894681, 0.24178252, 0.657233],
#         [0.00626879, 0.01704033, 0.04632042, 0.93037045],
#     ]
# )
# results.append(np.allclose(act.Softmax().f(X), Y))

X = np.array([1.0, 2.0, 3.0])
Y = np.array([0.09, 0.24, 0.67])
results.append(np.allclose(act.Softmax().f(X), Y, atol=1e-2))

X = np.array([1.0, 2.0, 5.0])
Y = np.array([0.02, 0.05, 0.93])
results.append(np.allclose(act.Softmax().f(X), Y, atol=1e-2))

X = np.array([1, 2])
Y = np.array([0.26894142, 0.73105858])
results.append(np.allclose(act.Softmax().f(X), Y))

print(f"Softmax works: {all(results)}")

# Test Softmax derivative
X = act.Softmax().f(np.array([1, 2]))
Y = np.array([[0.19661193, -0.19661193], [-0.19661193, 0.19661193]])
print(f"Softmax derivative works: {np.allclose(act.Softmax().df(X), Y)}")

# Make sure softmax output has the same shape as Sigmoid output
X = np.array([1, 2, 3])
Y_sigmoid = act.Sigmoid().f(X)
Y_softmax = act.Softmax().f(X)
print(f"Softmax output ({Y_softmax.shape}): {Y_softmax} [{Y_softmax.sum()}]")
print(f"Sigmoid output ({Y_sigmoid.shape}): {Y_sigmoid}")
print("")

# Softmax with layer
network = Network([Linear(3, 2, act.Softmax())])

X = np.array([[1, 2, 3]])
Y = np.array([[0.1, 0.2]])
network.train(X, Y)
