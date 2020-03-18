from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import tensorflow.keras.backend as K

class getEpochNo(Callback):
    def __init__(self):
        super(getEpochNo, self).__init__()
        self.epochNo = tf.Variable(1.)

    def on_epoch_end(self, epoch, logs={}):        
        K.set_value(self.epochNo, epoch + 1.0)

def Huber_Loss(y_true, y_pred):    
    xout = y_pred[:,:,:,0:2]
    return tf.keras.losses.Huber(delta=1.0, reduction=tf.keras.losses.Reduction.NONE)(xout, y_true)

def Max_Conf(epochCallback):
    def max_loss(y_true, y_pred):
        xout = y_pred[:,:,:,0:2]
        cout = y_pred[:,:,:,2][...,None]
        E1 = tf.keras.losses.Huber(delta=1.0, reduction=tf.keras.losses.Reduction.NONE)(xout, y_true)   
        E2 = 1 / epochCallback.epochNo * (cout - tf.multiply(cout,E1))
        return E2
    return max_loss

def All_Loss(epochCallback,  w1=1.0, w2=1.0):
    def loss(y_true, y_pred):        
        xout = y_pred[:,:,:,0:2]
        cout = y_pred[:,:,:,2][...,None]                
        
        E1 = w1 * tf.keras.losses.Huber(delta=1.0, reduction=tf.keras.losses.Reduction.NONE)(xout, y_true)   
        E2 = w2 / epochCallback.epochNo * (cout - tf.multiply(cout,E1))
        
        return E1 - E2
    return loss

def MSE_loss(y_true, y_pred):
    xout = y_pred[:,:,:,0:2]
    return tf.keras.losses.MSE(xout, y_true)

def MAE_loss(y_true, y_pred):
    xout = y_pred[:,:,:,0:2]
    return tf.keras.losses.MAE(xout, y_true)