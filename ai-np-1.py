#!/usr/bin/python3.5

import sys
import csv
import math
import numpy as np

class AI:
    weights = [
        np.random.sample((4, 9)),
        np.random.sample((3, 4)),
        np.random.sample((2, 3))
    ];
    layers = [ np.zeros(9), np.zeros(4), np.zeros(3), np.zeros(2) ]
    errors = [ np.zeros(9), np.zeros(4), np.zeros(3), np.zeros(2) ]

    data = []

    coef = 0.15

    def load(self, fn):
        with open(fn) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                cols = []
                for c in row: cols.append(float(c))
                self.data.append(cols)

    def __init__(self, fn):
        self.load(fn)

    def sigmoid(self, value):
        return 1 / (1 + math.exp(-value))

    def calculate_layer(self, layer, out, weights):
        for i in range(len(weights)): out[i] = self.sigmoid((weights[i] * layer).sum())
        return out

    def find_error(self, errors, weights, result):
        r = result.reshape((len(result), 1))
        errors = (weights * r).sum(axis=0)
        return errors

    def correction(self, weights, errors, data0, data1, coef):
        d0 = coef * errors * (1 - data0) * data0
        weights += d0.reshape((len(d0),1)) * data1
        return weights

    def calculate(self, data_input):
        self.layers[0] = data_input
        for i in range(1, len(self.layers)):
            self.layers[i] = self.calculate_layer(self.layers[i-1], self.layers[i], self.weights[i-1])

    def calculate_error(self, out):
        self.errors[-1] = out - self.layers[-1]
        for i in range(len(self.errors) - 2, 0, -1):
            self.errors[i] = self.find_error(self.errors[i], self.weights[i], self.errors[i + 1])
        error = 0
        for i in self.errors[-1]: error += i ** 2
        error /= len(self.errors[-1])
        return error

    def weight_correction(self):
        for i in range(len(self.weights)):
            self.weights[i] = self.correction(self.weights[i], self.errors[i+1], self.layers[i+1], self.layers[i], self.coef)

    def get_result(self):
        r = []
        for i in self.layers[-1]: r.append(round(i))
        return r

    def learning(self):
        error = 0
        for i in self.data:
            data_input = i[0:9]
            out = i[9:11]

            self.calculate(np.array(data_input))
            error += self.calculate_error(np.array(out))
            self.weight_correction()
        error /= len(self.data)
        return error

n = AI("data1.txt")
epoch = 0

while True:
    error = n.learning()
    print("\rEpoch: %6d, Error: %6.3f " % (epoch, error), end='')
    epoch += 1
    if error > 0.05: continue
    print()
    break

while True:
    print("Input: ", end='')
    i = input()
    n.calculate(list(map(float, i.split())))
    print(n.get_result())

