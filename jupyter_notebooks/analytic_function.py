import numpy as np


class Function:
    evaluate = ()
    differentiate = ()
    args = []
    
    def __init__(self, ev, dif):
        self.evaluate = ev
        self.differentiate = dif

    def __str__(self):
        r = [i.name for i in self.args].__str__()
        return r


def product(args):
    return np.prod(args)


def product_d(args):
    return [np.prod(args[:i] + args[i+1:]) for i in range(len(args))]


def funcProd(): return Function(product, product_d)


def exponential(args):
    return np.exp(args[0])


def exponential_d(args):
    return [exponential(args)]


def funcExp():return Function(exponential, exponential_d)


def sigmoid(args):
    x = args[0]
    return 1/(1 + np.exp(-x))


def sigmoid_d(args):
    return [sigmoid(args) * (1-sigmoid(args))]


def funcSig(): return Function(sigmoid, sigmoid_d)


def arctan(args):
    x = args[0]
    return np.arctan(x)


def arctan_d(args):
    x = args[0]
    return [1/(1 + x*x)]


def funcArcTan(): return Function(arctan, arctan_d)


def ReLU(args):
    x = args[0]
    return x if x>=0 else 0


def ReLU_d(args):
    x = args[0]
    return [1 if x>=0 else 0]


def funcReLU(): return Function(ReLU, ReLU_d)


def MatVec(args):
    return np.dot(args[0], args[1])


def MatVec_d(args):
    mat = args[0]
    vec = list(args[1])
    vec_d = mat
    mat_d = [[[0]*len(mat[0]) if j != i else vec for j in range(len(mat))] for i in range(len(vec))]
    return [mat_d, vec_d]


def funcMatVec(): return Function(MatVec, MatVec_d)