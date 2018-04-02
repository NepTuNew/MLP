import numpy as np
import math

class MLP:
    def __init__(self, lr, nInput, nHidden, nLayer, nOutput):
        """

        :param lr: learning rate
        :param nInput: the input dimension
        :param nHidden: the number of each hidden neurals
        :param nLayer: the layer that include output layer, so if 2 hidden and 1 output, then nLayer is 3
        :param nOutput: the output layer neural number(if class has 5 then output = 5), the ground truth is encoded to one_hot format
        """
        #MLP Weights initialize
        #first layer
        self.WList = [np.random.rand(nInput+1, nHidden)]
        #hidden layer
        for i in range(nLayer-2):
            self.WList.append(np.random.rand(nHidden+1, nHidden))
        #output layer
        self.WList.append(np.random.rand(nHidden+1, nOutput))

        self.nInput = nInput
        self.nHidden = nHidden
        self.nLayer = nLayer
        self.nOutput = nOutput
        self.layerOuts = []
        self.lr = lr
        self.sigmoid = np.vectorize(self.sigmoid)
        self.sigmoid_der = np.vectorize(self.sigmoid_der)

    def sigmoid(self, x):
        return float(1. / (1. + math.exp(-x)))

    # Sigmoid_der function not used in this project
    def sigmoid_der(self, x):
        return x*(1.-x)

    def forward(self, input):
        """

        :param input: input shape is (1, input_dimension) p.s not implement batch input
        :return: null
        """
        self.layerOuts = []
        for i in range(self.nLayer):
            if i == 0:
                value = np.dot(np.append([-1.], input), self.WList[0])
                self.layerOuts.append(input.reshape([1,-1]))
                value = self.sigmoid(value)
                self.layerOuts.append(value.reshape([1,-1]))
                continue
            else:
                value = np.dot(np.append([-1.], self.layerOuts[i]), self.WList[i])
            value = self.sigmoid(value)
            self.layerOuts.append(value.reshape([1,-1]))

    def backpropagate(self, output):
        """

        :param output: one_hot format ground truth, shape is (1, n_class)
        :return: null
        """

        #compute hidden-to-output layer delta
        deltas = []
        error = np.subtract(output.reshape([-1,1]), self.layerOuts[-1].reshape([-1,1]))#error = np.subtract(output.reshape([-1,1]), self.layerOuts[-1])
        der_last = np.multiply(self.layerOuts[-1].reshape([-1,1]), np.subtract(1, self.layerOuts[-1].reshape([-1,1])))
        delta_last =np.multiply(error, der_last)
        deltas.append(np.array(delta_last))

        #print('delta_last: {}'.format(delta_last))

        #compute hidden-to-hidden layer delta
        for i in range(self.nLayer-1):
            index = -(i + 2 ) #index from back
            #print(index)
            der_layer = np.multiply(self.layerOuts[index].reshape([-1,1]), np.subtract(1, self.layerOuts[index].reshape([-1,1]))) # index = j
            #print(self.WList[index+1][1:])
            error = np.dot(self.WList[index+1][1:], deltas[i]) # index = k location shape(3,)

            delta = np.multiply(der_layer, error)
            #print(delta)
            deltas.append(delta)

        #update each layer weights
        for i in range(self.nLayer):
            index = - (i + 1) # index from back
            a = np.dot(np.append([-1], self.layerOuts[index-1]).reshape([-1,1]), deltas[i].T)
            b = np.multiply(self.lr, a)
            self.WList[index] = np.add(self.WList[index], b)

    def precision(self, train, gt):
        total = len(train)
        correct = 0
        for i in range(total):
            self.forward(train[i])
            pred = self.layerOuts[-1]
            if pred.argmax() == gt[i].argmax():
                correct += 1
        print('====================')
        print('total: {}'.format(total))
        print('correct: {}'.format(correct))
        print('precision: {}'.format(correct/total))

        return correct/total

    def rmse(self, data, gt):
        rmse = 0.0
        for i in range(len(data)):
            self.forward(data[i])
            layerOuts = self.layerOuts[-1]
            error = np.subtract(layerOuts, gt[i])
            error_square = np.multiply(error, error)
            rmse += np.sum(error_square) / gt[0].shape[1]
        rmse /= len(data)
        return rmse





