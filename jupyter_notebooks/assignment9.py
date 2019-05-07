import jupyter_notebooks.back_propogation.backprop as bp
from jupyter_notebooks.back_propogation.backprop import Node as node
import jupyter_notebooks.back_propogation.analytic_function as af
import numpy as np
import matplotlib.pyplot as pyplot

size = [10, 25, 50]
iteration = [2000, 2000, 2000]
X = np.load('assignment8_X.npy')
Y = np.load('assignment8_Y.npy')
alpha = 0.07

x = node('x')
w1 = node('w1')
h1 = node('h1')
w2 = node('w2')
h2 = node('h2')
w3 = node('w3')
h3 = node('h3')

x.toNode(h1)
w1.toNode(h1)
h1.toNode(h2)
w2.toNode(h2)
h2.toNode(h3)
w3.toNode(h3)

h1.analyze(af.funcMatVec(), [w1, x], True)
h2.analyze(af.funcMatVec(), [w2, h1], True)
h3.analyze(af.funcMatVec(), [w3, h2], True)

'''
for index in range(3):
    m = size[index]
    ite = iteration[index]
    w1val = 0.01 * np.random.randn(m, 10)
    w2val = 0.01 * np.random.randn(m, m)
    w3val = 0.01 * np.random.randn(10, m)
    chain = []
    for i in range(ite):
        w1gradient = []
        w2gradient = []
        w3gradient = []
        loss = 0
        for j in range(len(X)):
            xval = X[j]
            yval = Y[j]
            bp.forward([x, w1, w2, w3], [xval, w1val, w2val, w3val])
            bp.backward(h3)
            w1gradient.append(sum([np.multiply((h3.value[j] - yval[j]), w1.der[j]) for j in range(len(yval))]))
            w2gradient.append(sum([np.multiply((h3.value[j] - yval[j]), w2.der[j]) for j in range(len(yval))]))
            w3gradient.append(sum([np.multiply((h3.value[j] - yval[j]), w3.der[j]) for j in range(len(yval))]))
            loss += np.sum([(h3.value[j] - yval[j]) ** 2 for j in range(len(yval))])
            bp.reset(h3)
        w1val -= alpha * sum(w1gradient) / 25 * 2
        w2val -= alpha * sum(w2gradient) / 25 * 2
        w3val -= alpha * sum(w3gradient) / 25 * 2
        chain.append(loss/25)
    print(loss/25)
    print(np.dot(np.dot(w3val, w2val), w1val))
    pyplot.subplots()
    pyplot.plot(chain)
    pyplot.show()
'''

x = node('x')
w1 = node('w1')
h1 = node('h1')
v1 = node('v1')
w2 = node('w2')
h2 = node('h2')
v2 = node('v2')
w3 = node('w3')
h3 = node('h3')

x.toNode(h1)
w1.toNode(h1)
h1.toNode(v1)
v1.toNode(h2)
w2.toNode(h2)
h2.toNode(v2)
v2.toNode(h3)
w3.toNode(h3)

h1.analyze(af.funcMatVec(), [w1, x], True)
v1.analyze(af.funcSig(), [h1], True)
h2.analyze(af.funcMatVec(), [w2, v1], True)
v2.analyze(af.funcSig(), [h2], True)
h3.analyze(af.funcMatVec(), [w3, v2], True)

rates = [0.5, 0.25, 0.25]

for index in range(2):
    m = size[index]
    ite = iteration[index]
    alpha = rates[index]
    w1val = 0.01 * np.random.randn(m, 10)
    w2val = 0.01 * np.random.randn(m, m)
    w3val = 0.01 * np.random.randn(10, m)
    chain = []
    for i in range(ite):
        w1gradient = []
        w2gradient = []
        w3gradient = []
        loss = 0
        for j in range(len(X)):
            xval = X[j]
            yval = Y[j]
            bp.forward([x, w1, w2, w3], [xval, w1val, w2val, w3val])
            bp.backward(h3)
            w1gradient.append(sum([np.multiply((h3.value[j] - yval[j]), w1.der[j]) for j in range(len(yval))]))
            w2gradient.append(sum([np.multiply((h3.value[j] - yval[j]), w2.der[j]) for j in range(len(yval))]))
            w3gradient.append(sum([np.multiply((h3.value[j] - yval[j]), w3.der[j]) for j in range(len(yval))]))
            loss += np.sum([(h3.value[j] - yval[j]) ** 2 for j in range(len(yval))])
            bp.reset(h3)
        w1val -= alpha * sum(w1gradient) / 25 * 2
        w2val -= alpha * sum(w2gradient) / 25 * 2
        w3val -= alpha * sum(w3gradient) / 25 * 2
        chain.append(loss)
    print(loss / 25)
    print(np.dot(np.dot(w3val, w2val), w1val))
    pyplot.subplots()
    pyplot.plot(chain)
    pyplot.show()


w = np.random.randn(10, 10)
X = np.load('assignment9_X.npy').transpose()
Y = np.load('assignment9_Y.npy')
delta = 1
alpha = 0.001
chain = []

for ite in range(500):
    gradient = 0
    loss = 0
    for i in range(len(X)):
        xval = X[i]
        yval = Y[i]

        def difference(k):
            result = np.dot(w[k], xval) - np.dot(w[yval], xval)
            return result

        wsum = (-sum([0 if j == yval else (1 if difference(j) + delta > 0 else 0) for j in range(10)]) * xval).tolist()
        local = [[0] * 10 if j != yval else wsum for j in range(10)]
        gradient = np.add(gradient, local)
        loss += sum([0 if j == yval else (0 if difference(j) - delta < 0 else difference(j) - delta) for j in range(10)])
    w -= alpha * gradient
    chain.append(loss)
print(loss)
pyplot.subplots()
pyplot.plot(chain)
pyplot.show()


