from __future__ import division, print_function
import theano
from theano import tensor as T

a = T.scalar()
b = T.scalar()
c = T.scalar()

y = a * b / c

print('a=%s' % a)
print('b=%s' % b)
print('c=%s' % c)
print('y', y, type(y), y.shape, y.dtype)

multiply = theano.function(inputs=[a, b, c], outputs=y)

d = multiply(2, 2, 2)
print('d', d, type(d), d.shape, d.dtype)

print(multiply(2, 2, 2)) # 2
print(multiply(3, 6, 2)) # 9
print(multiply(3, 3, 4)) # 2.25
