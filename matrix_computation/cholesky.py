import math
import numpy as np


def cholesky(M):
    n = len(M)
    r = [[0 for j in range(n)] for i in range(n)]
    for i in range(n):
        res = M[i][i] - (sum([r[x][i]*r[x][i] for x in range(i)]) if i > 0 else 0)
        if res <= 0 :
            raise ValueError("Given matrix is not positive definite")
        else:
            r[i][i] = math.sqrt(res)
        for j in range(i+1, n):
            r[i][j] = (M[i][j] - (sum([r[x][i]*r[x][j] for x in range(i)]) if i != 0 else 0))/r[i][i]
    return r


'''r = [[1,4,7,9],
     [0,2,1,6],
     [0,0,3,0],
     [0,0,0,4]]

r_t = [[r[j][i] for j in range(4)] for i in range(4)]

a = np.matmul(r_t, r)
print(a)

print(cholesky(a))
'''