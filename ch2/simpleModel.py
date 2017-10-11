#!/usr/bin/env python
# coding:utf-8

import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers
import matplotlib.pyplot as plt

def plotmodel(model):
	x_for_plot = Variable(np.array([[1],[2],[7]],dtype=np.float32))
	y_for_plot = model(x_for_plot)
	plt.plot(x_for_plot.data,y_for_plot.data,"r-")

#define model
model = L.Linear(1,1)
optimizer = optimizers.SGD()
optimizer.setup(model)

# number of times to study
times = 50

#input vector
x = Variable(np.array([[1],[2],[7]],dtype=np.float32))

#answer vector
t = Variable(np.array([[2],[4],[14]], dtype=np.float32))

for i in range(0,times):
    #inititalize gradient
    #optimizer.zero_grads()
    model.cleargrads()
    #predict
    y = model(x)

    #show output of the model
    print(y.data)

    #loss
    loss = F.mean_squared_error(y,t)

    #backward
    #print "model.W.grad before backward:", model.W.grad
    loss.backward()
    #print "model.W.grad after backward:", model.W.grad
    #update
    #print "model.W.data before update:", model.W.data
    optimizer.update()
    #print "model.W.data after update:", model.W.data
    #plotmodel(model)

print "result"
x = Variable(np.array([[3],[4],[5]], dtype=np.float32))
y = model(x)
print(y.data)
#plt.show()
