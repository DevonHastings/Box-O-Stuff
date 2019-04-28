"""
Basic Neural Network

~ Devon H.

                               +-------------+
                               | Data Format |
                               +-------------+
'x' is the data point
'y' is the label

( x, y ) = ( np.array([0, ..., s ]).reshape(s, 1), np.array([0, ..., t ]).reshape(t, 1) )

training_data = [(x_1, y_1), (x_2, y_2) ..., n]
test_data = [(x_1, y_1), (x_2, y_2) ..., n]

combined data --> (training_data, test_data)


"""

import numpy as np
import random


def sigmoid(array):
    return 1 / (1 + np.exp(-array))


def sigmoid_derivative(array):
    return sigmoid(array) * (1-sigmoid(array))


class Network(object):

    def __init__(self, layers):
        self.number_layers = len(layers)
        self.sizes = layers
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(layers[:-1], layers[1:])]

    def prediction(self, input_vector):
        a = input_vector
        for bias, weight in zip(self.biases, self.weights):
            a = sigmoid(np.dot(weight, a) + bias)
        return a

    def train(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):

        train_data_length = len(training_data)

        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, train_data_length, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(epoch, self.evaluate(test_data), len(test_data)))
            else:
                print("Epoch {0} complete".format(epoch))

    def update_mini_batch(self, mini_batch, learning_rate):
        # Clean slate for calculating partials
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            # Sum used used to approx gradient of C
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (learning_rate / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, data, label):
        # Clean slate for calculating partials
        nabla_b = [np.zeros(biases.shape) for biases in self.biases]
        nabla_w = [np.zeros(weights.shape) for weights in self.weights]

        activation = data  # setting the data equal to the first layers activation
        activations = [data]  # contain the activations from each layer
        zs = []  # contains the z-values ( i.e. z = w . a + b ) from each layer

        # forward pass
        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(weight, activation) + bias
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass

        # EQUATION 1 (Error for output layer)
        delta = self.cost_derivative(activations[-1], label) * sigmoid_derivative(zs[-1])

        # EQUATION 3 for last layer (del_C/del_b = error)
        nabla_b[-1] = delta

        # EQUATION 4 (del_C/del_w)
        # todo |                BIG QUESTION
        # todo | Why is each entry in nabla_w, (delta . activation)?
        # todo | The written equation has these two swapped.
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Finding delta for each layer then calculating (del_C/del_b) and
        # (del_C/del_w) for each layer

        for layer in range(2, self.number_layers):
            # EQUATION 2 (Error for any layer)
            delta = np.dot(self.weights[-layer+1].transpose(), delta) * sigmoid_derivative(zs[-layer])
            # EQUATION 3
            nabla_b[-layer] = delta

            # EQUATION 4 (del_C/del_w)
            # todo |                BIG QUESTION
            # todo | Why is each entry in nabla_w, (delta . activation)?
            # todo | The written equation has these two swapped.
            nabla_w[-layer] = np.dot(delta, activations[-layer-1].transpose())

        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.prediction(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    @staticmethod
    def cost_derivative(output_activation, label):
        return output_activation - label

    def return_weights(self):
        return self.weights

    def return_biases(self):
        return self.biases


train = [(np.array([1, 1]).reshape(2, 1), np.array([1]).reshape(1, 1)),
         (np.array([0, 1]).reshape(2, 1), np.array([0]).reshape(1, 1)),
         (np.array([1, 0]).reshape(2, 1), np.array([0]).reshape(1, 1)),
         (np.array([0, 0]).reshape(2, 1), np.array([1]).reshape(1, 1))]


test_1 = np.array([1, 1]).reshape(2, 1)
test_2 = np.array([0, 1]).reshape(2, 1)
test_3 = np.array([1, 0]).reshape(2, 1)
test_4 = np.array([0, 0]).reshape(2, 1)


""" =====NETWORK===== """
net = Network([2, 2, 1])
print(net.prediction(test_1))
print("hello")
""" ================= """


