import numpy as np
from jupyter_notebooks.back_propogation import backprop as bp, analytic_function as af
from jupyter_notebooks.back_propogation.backprop import Node as node

x = node('x')
#x.value = np.load('assignment7_X.npy')
w1 = node('w1')
#w1.value = np.load('assignment7_W1.npy')
w2 = node('w2')
#w2.value = np.load('assignment7_W2.npy')

h1 = node('h1')
h2 = node('h2')
f = node('f')

x.toNode(h1)
w1.toNode(h1)
h1.toNode(h2)
w2.toNode(h2)

h1.analyze(af.funcMatVec(), [w1, x], True)
h2.analyze(af.funcMatVec(), [w2, h1], True)

x_val = np.load('assignment7_X.npy')
w1_val = np.load('assignment7_W1.npy')
w2_val = np.load('assignment7_W2.npy')

bp.forward([x, w1, w2], [x_val, w1_val, w2_val])
print("value of h2")
print(h2.value)

bp.backward(h2)
print("\ngradient of W1")
print(w1.der)
print("\ngradient of W2")
print(w2.der)

print("\nnumerical gradient W2")
mat_w2 = w2.value
vec_h1 = h1.value
vec_h2 = h2.value
h = 0.001


def inc(i, j, mat):
    temp = (np.transpose(np.eye(1, len(mat_w2), i)) * np.eye(1, len(mat_w2[0]), j)) * h
    a = mat + temp
    return a


num_gradient_w2 = [[[(np.dot(inc(j, k, mat_w2), vec_h1)[i] - vec_h2[i])/h
                     for k in range(len(mat_w2[j]))]
                    for j in range(len(mat_w2))]
                   for i in range(len(vec_h1))]

print(num_gradient_w2)

print("\nnumerical gradient W1")


def inc_vec(i):
    a = [index for index in vec_h1]
    a[i] += h
    return a


mat_w1 = w1.value
vec_x = x.value
num_jacobian_h1 = [[(np.dot(mat_w2, inc_vec(j))[i] - vec_h2[i])/h for j in range(len(vec_h2))] for i in range(len(vec_h2))]
num_gradient_w1 = [[[(np.dot(inc(j, k, mat_w1), vec_x)[i] - vec_h1[i])/h
                     for k in range(len(mat_w1[0]))]
                    for j in range(len(mat_w1))]
                   for i in range(len(vec_x))]
num_gradient_w1 = np.dot(mat_w2, num_gradient_w1)
print(num_gradient_w1)