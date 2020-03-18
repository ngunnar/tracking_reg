import tensorflow as tf
import tfdeform
import math
import numpy as np

class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, heigth, width, batch_size, ds_size):
        self.input_dim = (heigth, width)
        self.tot_points = self.input_dim[0]*self.input_dim[1]
        self.batch_size = batch_size        
        self.ds_size = ds_size
        self.on_epoch_end()        
    
    def __len__(self):
        return math.ceil(self.ds_size / self.batch_size)
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate data
        inputs, outputs = self._get_train_samples()        
        return (inputs, outputs)  

    def _get_train_samples(self, s = None, d = None, points = None):
        if s is None:
            s = np.random.randint(40, 100)
        if d is None:
            d = np.random.randint(20)
        if points is None:
            points = np.random.randint(50) + 150
                
        dense = tfdeform.random_deformation_linear(
            shape=[self.batch_size, self.input_dim[0], self.input_dim[1]], std=s, distance=d)
        
        #noise = np.random.normal(loc=0.0, scale=d/10, size=(self.batch_size, self.input_dim[0], self.input_dim[1], 1))
        #dense_n = dense + noise 
        nums = np.zeros((self.batch_size, self.tot_points))
        nums[:,:points] = 1 
        [np.random.shuffle(x) for x in nums]        
        c = np.reshape(nums, (self.batch_size,self.input_dim[0], self.input_dim[1],1))        
        
        x = np.multiply(c, dense)
        #x_n = np.multiply(c, dense_n)
        #minumum = np.minimum(np.abs(x_n), np.abs(x))
        #maximum = np.maximum(np.abs(x_n), np.abs(x))
        #np.seterr(divide='ignore', invalid='ignore')
        #c_r = np.divide(minumum, maximum)
        #c_r = np.nan_to_num(c_r)
        #c_r = np.multiply(c_r, c)
        #return [x_n.astype(np.float32),c_r.astype(np.float32)] ,dense        
        return [x.astype(np.float32), c.astype(np.float32)], dense

    def on_epoch_end(self):
        'Updates indexes after each epoch'