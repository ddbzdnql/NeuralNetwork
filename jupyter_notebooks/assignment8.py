from jupyter_notebooks.back_propogation.backprop \
    import Node as node, backward as backward, forward as forward, reset as reset
import jupyter_notebooks.back_propogation.analytic_function as af
import numpy as np
import matplotlib.pyplot as pyplot

X = np.load('assignment8_X.npy')
Y = np.load('assignment8_Y.npy')
t = 25
ite = 2000
a = 0.01

# print(X.shape)
# print(Y.shape)

# structure of NN1

x = node('x')
w1 = node('w1')
w2 = node('w2')
h1 = node('h1')
h2 = node('h2')

x.toNode(h1)
w1.toNode(h1)
h1.toNode(h2)
w2.toNode(h2)

h1.analyze(af.funcMatVec(), [w1, x], True)
h2.analyze(af.funcMatVec(), [w2, h1], True)

'''
for m in [10, 25, 50]:
    w1val = 0.1 * np.random.randn(m, 10)
    w2val = 0.1 * np.random.randn(10, m)
    for index in range(ite):
        w1_gradient = []
        w2_gradient = []
        loss = 0
        for i in range(len(X)):
            xval = X[i]
            yval = Y[i]
            forward([x, w1, w2], [xval, w1val, w2val])
            backward(h2)
            w1_gradient.append(sum([np.multiply((h2.value[j] - yval[j]), w1.der[j]) for j in range(len(yval))]))
            w2_gradient.append(sum([np.multiply((h2.value[j] - yval[j]), w2.der[j]) for j in range(len(yval))]))
            loss += np.sum([(h2.value[j] - yval[j]) ** 2 for j in range(len(yval))])
            reset(h2)
        w1_g = sum(w1_gradient) / t * 2
        w2_g = sum(w2_gradient) / t * 2
        w1val -= a * w1_g
        w2val -= a * w2_g
    print(loss / t)
    print(np.dot(w2val, w1val))
    reset(h2)
'''

ite = 600
a = 0.055
#structure of NN2
x = node('x')
w1 = node('w1')
w2 = node('w2')
h1 = node('h1')
h2 = node('h2')
v1 = node('v1')

x.toNode(h1)
w1.toNode(h1)
h1.toNode(v1)
v1.toNode(h2)
w2.toNode(h2)

h1.analyze(af.funcMatVec(), [w1, x], True)
v1.analyze(af.funcSig(), [h1], True)
h2.analyze(af.funcMatVec(), [w2, v1], True)

for m in [10, 25, 50]:
    w1val = 0.1 * np.random.randn(m, 10)
    w2val = 0.1 * np.random.randn(10, m)
    for index in range(ite):
        w1_gradient = []
        w2_gradient = []
        loss = 0
        for i in range(len(X)):
            xval = X[i]
            yval = Y[i]
            forward([x, w1, w2], [xval, w1val, w2val])
            backward(h2)
            w1_gradient.append(sum([np.multiply((h2.value[j] - yval[j]), w1.der[j]) for j in range(len(yval))]))
            w2_gradient.append(sum([np.multiply((h2.value[j] - yval[j]), w2.der[j]) for j in range(len(yval))]))
            loss += np.sum([(h2.value[j] - yval[j]) ** 2 for j in range(len(yval))])
            reset(h2)
        w1_g = sum(w1_gradient) / t * 2
        w2_g = sum(w2_gradient) / t * 2
        w1val -= a * w1_g
        w2val -= a * w2_g
    print(loss / t)
    print(np.dot(w2val, w1val))
    reset(h2)
