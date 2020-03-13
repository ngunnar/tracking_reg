from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import tensorflow.keras.backend as K

class getEpochNo(Callback):
    def __init__(self):
        super(getEpochNo, self).__init__()
        self.epochNo = tf.Variable(1.)

    def on_epoch_end(self, epoch, logs={}):        
        K.set_value(self.epochNo, epoch + 1.0)
             

def HuberLoss(epochCallback):
    def loss(y_true, y_pred):    
        xout = y_pred[:,:,:,0][...,None]
        cout = y_pred[:,:,:,1][...,None]     
        E1 = tf.keras.losses.Huber(delta=1.0)(xout, y_true)   
        E2 = 1 / epochCallback.epochNo * (cout - cout*E1)
        return E1 - E2
    return loss