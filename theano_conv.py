import numpy as np
import theano
import theano.tensor as T
import time

x = T.tensor4('x')
filters = theano.shared(
    np.random.rand(256, 128, 3, 3).astype(theano.config.floatX),
    'filters')
# B x 1 x 1 x T
y = theano.gpuarray.dnn.dnn_conv(
    img=x,
    kerns=filters,
    border_mode='half',
    precision=theano.config.floatX)

f = theano.function([x], y)
x_ = np.random.rand(32, 128, 256, 256).astype(theano.config.floatX)
y_ = f(x_)
#f.sync_shared()
t0 = time.time()
for i in range(50):
    print i
    y_ = f(x_)
 #   f.sync_shared()
t1 = time.time()
print t1 - t0
