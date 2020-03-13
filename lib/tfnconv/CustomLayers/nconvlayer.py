from __future__ import absolute_import, division, print_function

from tensorflow.python.framework import tensor_shape
from tensorflow.keras.layers import Layer
import tensorflow as tf
from .propconfidenslayer import PropConf
from .Utils import softplus

class Nconv(Layer):
    def __init__(self, filters, kernel_size, name, strides = 1, padding="SAME", dilation=1, bias_intializer='zeros', **kwargs):
        super(Nconv, self).__init__(name = name, **kwargs)
        self.eps = 1e-20
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilations = dilation
        self.padding = padding
        self.kernel_initializer = "he_normal" 
        self.bias_initializer = "zeros"
        self.beta = 10

    def build(self, input_shape):        
        input_shape = tensor_shape.TensorShape(input_shape[0])
        input_channel = int(input_shape[-1])
        kernel_shape = self.kernel_size + (input_channel, self.filters)        
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=None,
                                      constraint=None)
        
        self.bias = self.add_weight(
            name='bias',
            shape=(self.filters,),
            initializer='zeros',
            trainable=True,
            dtype=self.dtype)
        
        self.conf_layer = PropConf(self.beta)   
        self.built = True
    
    def call(self, inputs):        
        data = inputs[0]
        conf = inputs[1]

        denom = tf.nn.conv2d(conf, softplus(self.beta, self.kernel), strides=self.strides, padding=self.padding, dilations = self.dilations)
        nomin = tf.nn.conv2d(data*conf, softplus(self.beta, self.kernel), strides=self.strides, padding=self.padding, dilations = self.dilations)
        
        nconv = nomin / (denom+self.eps)        
        nconv = tf.nn.bias_add(nconv, self.bias)            
        cout = self.conf_layer([denom, self.kernel])        
        return nconv, cout

    def get_config(self):
        config = {
            'eps': self.eps,
            'filters': self.filters,
            'kernel_size':self.kernel_size,
            'strides':self.strides,
            'dilations':self.dilations,
            'padding': self.padding,
            'kernel_initializer':self.kernel_initializer,
            'beta':self.beta,
            'bias_initializer': self.bias_initializer,
            }
        base_config = super(Nconv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))