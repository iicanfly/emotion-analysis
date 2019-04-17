#coding:utf-8
import json
import random
import sys

import io
reload(sys) 
sys.setdefaultencoding('utf-8')

# Third-party libraries
import numpy as np
np.set_printoptions(threshold='nan')


#### Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.
        """
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        return (a - y)


#### Main Network class
class Network(object):

    def __init__(self, sizes, cost = CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        count = 0
        for b, w in zip(self.biases, self.weights):
            count = count + 1
            if (count < len(self.sizes) - 1):
                a = relu(np.dot(w, a) + b)
            else:
                a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, accuracyValue,
            lmbda = 0.0,
            predictData = None,
            evaluation_data = None,
            monitor_evaluation_cost = False,
            monitor_evaluation_accuracy = False,
            monitor_training_cost = False,
            monitor_training_accuracy = False):
        if evaluation_data: 
            n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size]  for k in range(0, n, mini_batch_size)]
            
            print("Epoch %s training now" % j)
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            print("Epoch %s training complete" % j)
           
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data))
                accuracyTmp = accuracy / n_data
# 				if accuracyTmp > 0.873:
                accuracyValue[0] = accuracyTmp
                accuracyValue[1] = accuracyValue[1] + 1
                #self.outPredict(predictData, accuracyValue)
                self.save(j)
        # return evaluation_cost, evaluation_accuracy, \
        #        training_cost, training_accuracy
        return accuracyValue

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        count = 0
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            count = count + 1
            if (count < len(self.sizes) - 1):
                activation = relu(z)
            else:
                activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = relu_prime(z)
            delta = np.multiply(np.dot(self.weights[-l + 1].transpose(), delta), sp)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert = False):
        results = []
        sum = 0
        # if convert:
        # 	results = [(self.feedforward(x), y)
        # 	           for (x, y) in data]
        # else:
        for (x, y) in data:
            mat1 = matrixRound(self.feedforward(x))
            results.append((mat1, y))
            if (mat1 == y):
                sum = sum + 1
        return sum

    # 输出预测集结果
    def outPredict(self, preData, accuracyValue):
        ofstream = io.open("./result/result%d.txt" % (accuracyValue[1]), 'w', encoding = 'UTF-8')
        ofstream.write(u"Accuracy: " + (str(accuracyValue[0]).decode('utf-8')))
        ofstream.write(u'\n')
        for data in preData:
            matResult = matrixRound(self.feedforward(data))
            value=int(matResult[0][0])
            ofstream.write(str(value).decode('utf-8'))
            ofstream.write(u'\n')
        ofstream.close()
        return

    def total_cost(self, data, lmbda, convert = False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)
        cost += 0.5 * (lmbda / len(data)) * sum(
            np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def save(self, j):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": self.cost.__name__.decode('utf-8')}
        
        f = io.open("./parameter/epoch%s_sizes.txt" % j, "wb")
        json.dump(data["sizes"],f)
        f.close()
        f = io.open("./parameter/epoch%s_weights.txt" % j, "wb")
        json.dump(data["weights"],f)
        f.close()
        f = io.open("./parameter/epoch%s_biases.txt" % j, "wb")
        json.dump(data["biases"],f)
        f.close()
        f = io.open("./parameter/epoch%s_cost.txt" % j, "wb")
        json.dump(data["cost"],f)
        f.close()


#### Loading a Network
def load(filename_sizes, filename_weights, filename_biases, filename_cost):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    """
    f = io.open(filename_weights, "r")
    data_weights = json.load(f)
    f.close()
    
    f = io.open(filename_sizes, "r")
    data_sizes = json.load(f)
    f.close()
    
    f = io.open(filename_biases, "r")
    data_biases = json.load(f)
    f.close()
    
    f = io.open(filename_cost, "r")
    data_cost = json.load(f)
    f.close()
    
    cost = getattr(sys.modules[__name__], data_cost)
    net = Network(data_sizes, cost = cost)
    net.weights = [np.array(w) for w in data_weights]
    net.biases = [np.array(b) for b in data_biases]
    return net


#### Miscellaneous functions
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


def relu(a):
    return np.maximum(a, np.zeros(a.shape).astype(a.dtype))


def relu_prime(a):
    return (a > 0.).astype(a.dtype)


def leakRelu(x, alpha = 0.01):
    return np.maximum(alpha * x, x)


def leakRelu_prime(x, alpha = 0.01):
    dx = np.ones_like(x)
    dx[x <= 0] = alpha
    return dx


def softmax(a):
    mat1 = np.exp(a - np.max(a, axis = 0))
    mat2 = np.sum(np.exp(a - np.max(a, axis = 0)), axis = 0)
    return np.divide(mat1, mat2)



# 对矩阵取整函数
def matrixRound(M):
    # 对行循环
    for index in range(len(M)):
        # 对列循环
        for _index in range(len(M[index])):
            M[index][_index] = round(float(M[index][_index]))
    return M
