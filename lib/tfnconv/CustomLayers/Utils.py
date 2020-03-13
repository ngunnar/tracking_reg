from __future__ import absolute_import, division, print_function

import tensorflow as tf

def softplus(beta, x):
    return 1/beta * tf.nn.softplus(beta*x)