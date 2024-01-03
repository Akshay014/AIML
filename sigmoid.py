import numpy as np

X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

# scale units
X /= np.amax(X, axis=0)  # maximum of X array
y /= 100  # maximum test score is 100


class NeuralNetwork(object):
    def _init(self, learning_rate=0.01):  # Corrected method name to __init_
        # parameters
        self.inputSize, self.outputSize, self.hiddenSize = 2, 1, 3
        # weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)
        # learning rate
        self.learning_rate = learning_rate

    def feedForward(self, X):
        # forward propagation through the network
        self.z2 = self.sigmoid(np.dot(X, self.W1))  # activation function
        output = self.sigmoid(np.dot(self.z2, self.W2))
        return output

    def sigmoid(self, s, deriv=False):
        if deriv:
            return s * (1 - s)
        return 1 / (1 + np.exp(-s))

    def backward(self, X, y, output):
        # backward propagate through the network
        self.output_delta = (y - output) * self.sigmoid(output, deriv=True)
        self.z2_delta = self.output_delta.dot(self.W2.T) * self.sigmoid(self.z2, deriv=True)
        # adjust weights with learning rate
        self.W1 += self.learning_rate * X.T.dot(self.z2_delta)
        self.W2 += self.learning_rate * self.z2.T.dot(self.output_delta)

    def train(self, X, y, epochs=100):
        for i in range(epochs):
            Loss = np.mean(np.square(y - self.feedForward(X)))
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (i, self.learning_rate, Loss))
            self.backward(X, y, self.feedForward(X))


# Instantiate the neural network with a learning rate
NN = NeuralNetwork(learning_rate=0.01)
# Train the neural network
NN.train(X, y)
print("Input:", X)
print("Actual Output:", y)
print("Loss:", np.mean(np.square(y - NN.feedForward(X))))
print("\nPredicted Output:", NN.feedForward(X))
