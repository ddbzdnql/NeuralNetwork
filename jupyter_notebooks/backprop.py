import numpy as np


class Node:

    def __init__(self, n):
        self.name = n
        self.value = 0.0
        self.to = []
        self.feed = []
        self.analysis = []
        self.ready = False
        self.done = False
        
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
                for i in range(len(func.args)):
                    n = func.args[i]
                    try:
                        result = current.der * (gradient[i] if sign else -gradient[i])
                    except TypeError:
                        print(n.name + current.name)
                        result = np.dot(current.der, (gradient[i] if sign else -gradient[i]))
                    if hasattr(n, 'der'):
                        n.der += result
                    else:
                        n.der = result
            current.done = True

            for n in current.feed:
                if n not in nodes:
                    nodes.append(n)
