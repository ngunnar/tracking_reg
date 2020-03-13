from __future__ import absolute_import, division, print_function

from tensorflow.keras.layers import Layer
from .Utils import softplus
import tensorflow as tf

class PropConf(Layer):
    def __init__(self, beta):
        super(PropConf, self).__init__()
        self.beta = beta
    
    def build(self, input_shape):
        super(PropConf, self).build(input_shape)
        #self.trainable = False       
        
    def call(self, inputs):
        cout = inputs[0]
        weights = inputs[1]        
        return cout / tf.math.reduce_sum(softplus(self.beta, weights))

    def get_config(self):
        config = {
            'beta':self.beta,            
            }
        base_config = super(PropConf, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))