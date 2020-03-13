from __future__ import absolute_import, division, print_function

from tensorflow.keras.layers import Layer, Flatten, Reshape
import tensorflow as tf

class Downsample(Layer):
    def __init__(self, input_shape, name):
        self.flatten = Flatten()
        self.reshape = Reshape((input_shape[1]//2, input_shape[2]//2, input_shape[3]))
        super(Downsample, self).__init__(name = name)
    
    def call(self, inputs):
        x = inputs[0]
        c = inputs[1]        
        c_ds, idx = tf.nn.max_pool_with_argmax(input=c, ksize=2, strides=2, padding='SAME')
        x_flatten = self.flatten(x)
        idx_flatten = self.flatten(idx)
        
        x_dss = self.reshape(tf.gather(x_flatten, idx_flatten, batch_dims=1, name=None))
        return x_dss, c_ds/4

    def get_config(self):
        config = {
            'input_shape': self.input_shape,
            }
        base_config = super(Downsample, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))        