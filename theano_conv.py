import numpy as np
import theano
import theano.tensor as T
import time

x = T.tensor4('x')
filters = theano.shared(
    np.random.rand(256, 256, 9, 9).astype(theano.config.floatX),
    'filters')
# B x 1 x 1 x T
y = T.nnet.conv2d(
    input=x,
    filters=filters,
    border_mode='half',
    filter_flip=False)

f = theano.function([x], y)
x_ = np.random.rand(256, 256, 70, 70).astype(theano.config.floatX)
y_ = f(x_)
f.sync_shared()
t0 = time.time()
for i in range(10):
    print i
    y_ = f(x_)
    f.sync_shared()
t1 = time.time()
print t1 - t0
