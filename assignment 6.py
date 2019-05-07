from jupyter_notebooks.back_propogation.backprop import Node as node
from jupyter_notebooks.back_propogation import backprop as bp, analytic_function as af

print("assignment 6\n")
print("graph 1:")
x = node('x')
y = node('y')
z = node('z')
v1 = node('v1')
v2 = node('v2')
v3 = node('v3')
v4 = node('v4')
v5 = node('v5')
x.toNodes([v1, v2])
y.toNodes([v1, v2])
z.toNodes([v1, v2])
v1.toNode(v3)
v2.toNode(v4)
v3.toNode(v5)
v4.toNode(v5)
v1.analyze(af.funcProd(), [x, y], True)
v1.analyze(af.funcProd(), [z], True)
v2.analyze(af.funcProd(), [x], True)
v2.analyze(af.funcProd(), [y, z], False)
v3.analyze(af.funcProd(), [v1, v1], True)
v4.analyze(af.funcExp(), [v2], True)
v5.analyze(af.funcProd(), [v3], True)
v5.analyze(af.funcProd(), [v4], True)

initVals = [1,2,0]
bp.forward([x,y,z], initVals)
bp.backward(v5)
graph = [x, y, z, v1, v2, v3, v4, v5]
for i in graph:
    print(i.name + " value: " + i.value.__str__() + ", der: " + i.der.__str__())

print("\ngraph 2")

x = node('x')
y = node('y')
z = node('z')
v1 = node('v1')
v2 = node('v2')
v3 = node('v3')
v4 = node('v4')
v5 = node('v5')
x.toNodes([v1, v2])
y.toNodes([v1, v2])
z.toNodes([v1, v2])
v1.toNode(v3)
v2.toNode(v4)
v3.toNode(v5)
v4.toNode(v5)
v1.analyze(af.funcProd(), [x, y], True)
v1.analyze(af.funcProd(), [x, z], True)
v2.analyze(af.funcProd(), [x, z], True)
v2.analyze(af.funcProd(), [y, z], False)
v3.analyze(af.funcSig(), [v1], True)
v4.analyze(af.funcArcTan(), [v2], True)
v5.analyze(af.funcProd(), [v3, v4], True)

initVals = [1, 1, 1]
bp.forward([x, y, z], initVals)
bp.backward(v5)
graph = [x, y, z, v1, v2, v3, v4, v5]
for i in graph:
    print(i.name + " value: " + i.value.__str__() + ", der: " + i.der.__str__())

print("\ngraph 3")
x = node('x')
y = node('y')
v1 = node('v1')
v2 = node('v2')
v3 = node('v3')
v4 = node('v4')
v5 = node('v5')
v6 = node('v6')
v7 = node('v7')
v8 = node('v8')
v9 = node('v9')

x.toNodes([v1, v2])
y.toNodes([v1, v2])
v1.toNodes([v3])
v2.toNodes([v4])
v3.toNodes([v5, v6])
v4.toNodes([v5, v6])
v5.toNodes([v7])
v6.toNodes([v8])
v7.toNodes([v9])
v8.toNodes([v9])

v1.analyze(af.funcProd(), [x], True)
v1.analyze(af.funcProd(), [x], True)
v1.analyze(af.funcProd(), [y], True)
v2.analyze(af.funcProd(), [x], True)
v2.analyze(af.funcProd(), [y], False)
v2.analyze(af.funcProd(), [y], False)
v3.analyze(af.funcSig(), [v1], True)
v4.analyze(af.funcSig(), [v2], True)
v5.analyze(af.funcProd(), [v3], True)
v5.analyze(af.funcProd(), [v4], False)
v6.analyze(af.funcProd(), [v3], True)
v6.analyze(af.funcProd(), [v4], False)
v7.analyze(af.funcSig(), [v5], True)
v8.analyze(af.funcSig(), [v6], True)
v9.analyze(af.funcProd(), [v7], True)
v9.analyze(af.funcProd(), [v8], True)

initVals = [1, 1]
bp.forward([x, y], initVals)
bp.backward(v9)
graph = [x, y, v1, v2, v3, v4, v5, v6, v7, v8, v9]
for i in graph:
    print(i.name + " value: " + i.value.__str__() + ", der: " + i.der.__str__())

print("\ngraph 4")
x = node('x')
y = node('y')
v1 = node('v1')
v2 = node('v2')
v3 = node('v3')
v4 = node('v4')
v5 = node('v5')
v6 = node('v6')
v7 = node('v7')
v8 = node('v8')
v9 = node('v9')

x.toNodes([v1, v2])
y.toNodes([v1, v2])
v1.toNodes([v3])
v2.toNodes([v4])
v3.toNodes([v5, v6])
v4.toNodes([v5, v6])
v5.toNodes([v7])
v6.toNodes([v8])
v7.toNodes([v9])
v8.toNodes([v9])

v1.analyze(af.funcProd(), [x], True)
v1.analyze(af.funcProd(), [x], True)
v1.analyze(af.funcProd(), [y], True)
v2.analyze(af.funcProd(), [x], True)
v2.analyze(af.funcProd(), [y], False)
v2.analyze(af.funcProd(), [y], False)
v3.analyze(af.funcReLU(), [v1], True)
v4.analyze(af.funcReLU(), [v2], True)
v5.analyze(af.funcProd(), [v3], True)
v5.analyze(af.funcProd(), [v4], False)
v6.analyze(af.funcProd(), [v3], True)
v6.analyze(af.funcProd(), [v4], False)
v7.analyze(af.funcReLU(), [v5], True)
v8.analyze(af.funcReLU(), [v6], True)
v9.analyze(af.funcProd(), [v7], True)
v9.analyze(af.funcProd(), [v8], True)

initVals = [1, 1]
bp.forward([x, y], initVals)
bp.backward(v9)
graph = [x, y, v1, v2, v3, v4, v5, v6, v7, v8, v9]
for i in graph:
    print(i.name + " value: " + i.value.__str__() + ", der: " + i.der.__str__())