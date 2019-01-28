#!/usr/bin/python3.5

import sys
import csv
import math
import getopt
import numpy as np

class AI:
    w_hl0 = np.random.sample((4, 9))
    w_hl1 = np.random.sample((3, 4))
    w_out = np.random.sample((2, 3))
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
        self.layers[1] = self.calculate_layer(self.layers[0], self.layers[1], self.w_hl0)
        self.layers[2] = self.calculate_layer(self.layers[1], self.layers[2], self.w_hl1)
        self.layers[3] = self.calculate_layer(self.layers[2], self.layers[3], self.w_out)

    def normalize_answer(self, value):
        return 0.2 + value * 0.6

    def calculate_error(self, out):
        self.errors[3] = out - self.layers[3]
        self.errors[2] = self.find_error(self.errors[2], self.w_out, self.errors[3])
        self.errors[1] = self.find_error(self.errors[1], self.w_hl1, self.errors[2])

        error = 0
        for i in self.errors[3]: error += i ** 2
        error /= len(self.errors[3])
        return error

    def weight_correction(self):
        self.w_hl0 = self.correction(self.w_hl0, self.errors[1], self.layers[1], self.layers[0], self.coef)
        self.w_hl1 = self.correction(self.w_hl1, self.errors[2], self.layers[2], self.layers[1], self.coef)
        self.w_out = self.correction(self.w_out, self.errors[3], self.layers[3], self.layers[2], self.coef)

    def get_result(self):
        r = []
        for i in self.layers[3]: r.append(round(i))
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
    print("Input: ")
    i = input()
    n.calculate(list(map(float, i.split())))
    print(n.get_result())

