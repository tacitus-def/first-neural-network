#!/usr/bin/python3.5

import sys
import csv
import math
import numpy as np

hline = np.ones((3), int)
vline = hline.copy()
vline.transpose()

for x in range(4):
    for y in range(4):
        matrix = np.zeros((3, 3), int)
        if x < 3: matrix[x] = hline
        if y < 3: matrix[:,y] = vline
        if x == 3 and y == 3: break
        row = list(matrix.reshape(9))
        row.append(1 if x < 3 else 0)
        row.append(1 if y < 3 else 0)
        print (row)
