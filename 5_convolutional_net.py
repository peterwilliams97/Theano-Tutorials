from __future__ import division, print_function
"""
    This implementation gives 99.5% accuracy on MNIST

    See benchmarks in http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html
"""
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

theano.config.floatX = 'float32'
theano.config.openmp = True
print(theano.config)


srng = RandomStreams()


def floatX(X):

    return np.asarray(X, dtype=theano.config.floatX)


def init_weights(shape):
    w0 = floatX(np.random.randn(*shape) * 0.01)
    w1 = theano.shared(w0)
    print('init_weights:w0=%s,w1=%s' % (S(w0), S(w1)))
    return w1


def rectify(X):
    return T.maximum(X, 0.0)


def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


def dropout(X, p=0.0):
    if p > 0.0:
        retain_prob = 1.0 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X


def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


def S(t):
    try:
        a = t.get_value()
    except:
        a = t
    return '%s.%s' % (list(a.shape), a.dtype)


def model(X, w1, w2, w3, w4, p_drop_conv, p_drop_hidden):
    print('model:X=%s,w1=%s' % (S(X), S(w1)))
    print('model:p_drop_conv%.3f,p_drop_hidden=%.3f' % (p_drop_conv, p_drop_hidden))
    l1a = rectify(conv2d(X, w1, border_mode='full'))
    l1 = max_pool_2d(l1a, (2, 2))
    l1 = dropout(l1, p_drop_conv)

    l2a = rectify(conv2d(l1, w2))
    l2 = max_pool_2d(l2a, (2, 2))
    l2 = dropout(l2, p_drop_conv)

    l3a = rectify(conv2d(l2, w3))
    l3b = max_pool_2d(l3a, (2, 2))
    l3 = T.flatten(l3b, outdim=2)
    l3 = dropout(l3, p_drop_conv)

    l4 = rectify(T.dot(l3, w4))
    l4 = dropout(l4, p_drop_hidden)

    pyx = softmax(T.dot(l4, w_o))
    return l1, l2, l3, l4, pyx

trX, teX, trY, teY = mnist(onehot=True)

trX = trX.reshape(-1, 1, 28, 28)
teX = teX.reshape(-1, 1, 28, 28)

X = T.ftensor4()
Y = T.fmatrix()

w1 = init_weights(( 32,  1, 3, 3))
w2 = init_weights(( 64, 32, 3, 3))
w3 = init_weights((128, 64, 3, 3))
w4 = init_weights((128 * 3 * 3, 625))
w_o = init_weights((625, 10))

noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w1, w2, w3, w4, 0.2, 0.5)
l1, l2, l3, l4, py_x = model(X, w1, w2, w3, w4, 0., 0.)
y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w1, w2, w3, w4, w_o]
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)


def score_str(score):
    return '%.4f %7.4f %.3f' % (score, 1.0 - score, -np.log10(np.abs(1.0 - score)))


score_list = []

for i in range(1000):
    best_score, best_i = max(score_list) if score_list else (-1.0, -1)
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    score = np.mean(np.argmax(teY, axis=1) == predict(teX))
    is_best = '***' if score > best_score else ''
    print('%3d: %s %s' % (i, score_str(score), is_best))
    if score <= best_score and i >= max(50, best_i + 40):
        break
    score_list.append((score, i))

print('-' * 80)
print('best scores')
score_list.sort(key=lambda x: (-x[0], x[1]))
for score, i in score_list[:10]:
    print('%3d: %s' % (i, score_str(score)))
print('-' * 80)
