import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import io, os
import numpy as np
from tf_Utils import draw_hsv

def Saver(name, at_epoch):    
    class CustomSaver(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (epoch % at_epoch) == 0:  # or save after some epoch, each k-th epoch etc.
                self.model.save_weights("{}.h5".format(name))                
    return CustomSaver()

def Tensorboard_callback(log_dir, ds, model):
    class CustomTensorBoard(TensorBoard):
        def __init__(self, **kwargs):  # add other arguments to __init__ if you need
            super().__init__(**kwargs)
        
        def on_epoch_end(self, epoch, logs={}):
            
            def splt(fig, axs, conf, pred, gt):                
                i = 0
                axs[i].title.set_text('Prediction')
                im1 = axs[i].imshow(draw_hsv(pred[None,...])[0,...])
                axs[i].axis('off')
                
                i = 1
                axs[i].title.set_text('Truth')                
                im2 = axs[i].imshow(draw_hsv(gt[None,...])[0,...])
                axs[i].contour(conf)
                axs[i].axis('off')   
                
            
            def _get_img(epoch):
                inputs1, outputs1 = ds._get_train_samples(s=1,d=1,points=200)
                pred1 = model(inputs1)
                inputs2, outputs2 = ds._get_train_samples(s=20,d=10,points=200)
                pred2 = model(inputs2)                
                inputs3, outputs3 = ds._get_train_samples(s=120,d=20,points=200)
                pred3 = model(inputs3)
                
                #figure, axs = plt.subplots(3, 3, figsize=(6, 9))
                fig =  plt.figure(figsize=(6, 9))
                fig.set_figwidth(15)    
                
                axs = [fig.add_subplot(3,2,1), fig.add_subplot(3,2,2)]                
                splt(fig, axs, inputs1[1][0,:,:,0], pred1[0,...], outputs1[0,...])                
                axs = [fig.add_subplot(3,2,3), fig.add_subplot(3,2,4)]
                splt(fig, axs, inputs2[1][0,:,:,0], pred2[0,...], outputs2[0,...])                
                axs = [fig.add_subplot(3,2,5), fig.add_subplot(3,2,6)]
                splt(fig, axs, inputs3[1][0,:,:,0], pred3[0,...], outputs3[0,...]) 
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                # Closing the figure prevents it from being displayed directly inside
                # the notebook.
                plt.close(fig)
                buf.seek(0)
                # Convert PNG buffer to TF image
                image = tf.image.decode_png(buf.getvalue(), channels=4)
                # Add the batch dimension
                image = tf.expand_dims(image, 0)            
                return image
            
            img = _get_img(epoch)            
            
            with file_writer.as_default():               
                tf.summary.image("Estimation", img, step=epoch)                
            super().on_epoch_end(epoch, logs)
    
    # Tensorboard
    logdir = os.path.join("logs", log_dir)
    file_writer = tf.summary.create_file_writer(logdir + '/img')
    return CustomTensorBoard(log_dir= logdir, 
                       histogram_freq=1,
                       profile_batch = 0,
                       embeddings_freq=0,
                       write_grads=False)