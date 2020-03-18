
from .model import create_model
from .losses import getEpochNo, All_Loss, Huber_Loss, MSE_loss, MAE_loss, Max_Conf
from .data_generator import CustomDataGenerator

import tensorflow as tf
import os
from datetime import datetime

def train_model(name):
    height = 256
    width = 256
    batch_size = 8
    ds_size = 8 * 5
    epochs = 200
    m = create_model("SAME", height, width, name)
    instance = getEpochNo()
    log_dir = "{0}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))

    m.summary()
    m.compile(loss = All_Loss(instance, w1=1.0, w2=1.0), 
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
              metrics=[Huber_Loss, MSE_loss, MAE_loss, Max_Conf(instance)])

    ds = CustomDataGenerator(height, width, batch_size, ds_size)
    tensorboard_callback = Tensorboard_callback(log_dir, ds, m)
    saver = CustomSaver(name, 20)
    m.fit_generator(ds, epochs = epochs, callbacks = [instance, tensorboard_callback, saver])
    m.save_weights('{0}.h5'.format(name))