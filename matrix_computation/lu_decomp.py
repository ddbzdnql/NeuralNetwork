import math
import numpy as np


def LU(matrix):
    m = matrix
    for i in range(len(m) - 1):
        current = m[i][i]
        if m[i][i] == 0:
            raise ValueError("Not singular matrix")
        for j in range(i+1, len(m)):
            m[j][i] = m[j][i] / m[i][i]
            for k in range(i+1, len(m[j])):
                m[j][k] -= m[j][i] * m[i][k]
        print(m)
    L = [[1 if i == j else 0 if i<=j else m[i][j] for j in range(len(m[0]))] for i in range(len(m))]
    U = [[0 if i > j else m[i][j] for j in range(len(m[0]))] for i in range(len(m))]
    return L, U


'''A = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

L, U = LU(A)
print((L, U))
print(np.matmul(L, U))
'''
