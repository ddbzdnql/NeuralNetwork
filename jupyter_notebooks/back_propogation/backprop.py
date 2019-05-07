import numpy as np
import time

class Node:

    def __init__(self, n):
        self.name = n
        self.value = None
        self.to = []
        self.feed = []
        self.analysis = []
        self.ready = False
        self.done = False
        self.der = None
        
    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        r = "" + self.name + ":\n" + \
            "\t" + [i.name for i in self.to].__str__() + "\n" +\
            "\t" + [i.name for i in self.feed].__str__() + "\n"
        for i,_ in self.analysis:
            r += "\t" + i.__str__() + "\n"
        return r

    def toNode(self, n):
        self.to.append(n)
        n.feed.append(self)

    def toNodes(self, n):
        for i in n:
            self.toNode(i)

    def analyze(self, func, args, sign):
        func.args = args
        self.analysis.append((func, sign))


def forward(nodes, initVals):
    while len(nodes) != 0:
        current = nodes.pop(0)
        prepared = True
        for n in current.feed:
            if not n.ready:
                prepared = False
                break
        
        if prepared:
            val = 0
            if len(initVals) > 0:
                val = initVals.pop(0)
            else:
                for func, sign in current.analysis:
                    params = [i.value for i in func.args]
                    temp = func.evaluate(params)
                    val += temp if sign else -temp
            current.value = val
            current.ready = True
            for n in current.to:
                if not (n in nodes):
                    nodes.append(n)
                    
    
def backward(node):
    node.der = 1
    nodes = [node]
    while len(nodes) != 0:
        current = nodes.pop(0)
        prepared = True
        for n in current.to:
            if not n.done:
                prepared = False
                break

        if prepared:
            for func, sign in current.analysis:
                params = [i.value for i in func.args]
                gradient = func.differentiate(params)
                cshape = np.shape(current.der)
                for i in range(len(func.args)):
                    n = func.args[i]
                    nshape = np.shape(gradient[i])
                    if len(cshape) == 2:
                        if len(nshape) == 3:
                            result = [sum([head[part] * np.array((gradient[i][part] if sign else -gradient[i][part]))
                                           for part in range(len(head))]) for head in current.der]
                        else:
                            result = np.dot(current.der, gradient[i] if sign else - gradient[i])
                    else:
                        result = current.der * gradient[i]
                    if not isinstance(n.der, type(None)):
                        n.der += result
                    else:
                        n.der = result
            current.done = True

            for n in current.feed:
                if n not in nodes:
                    nodes.append(n)


def reset(tail):
    stack = [tail]
    while len(stack) != 0:
        current = stack.pop(-1)
        current.value = 0.0
        current.ready = False
        current.done = False
        current.der = None
        for i in current.feed:
            stack.append(i)
