import theano
import theano.tensor as T
import numpy as np
import time

x_1 = T.fmatrix("x_1")
x_2 = T.fmatrix("x_2")
y = T.dot(x_1, x_2)

f = theano.function([x_1, x_2], y)
x_1_ = np.random.rand(256, 256).astype(theano.config.floatX)
x_2_ = np.random.rand(256, 256).astype(theano.config.floatX)

y_ = f(x_1_, x_2_)
t0 = time.time()
for i in range(5000):
    if i % 100 == 0:
        print i
    y_ = f(x_1_, x_2_)

t1 = time.time()
print t1 - t0

