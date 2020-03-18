import lib.tfnconv.CustomLayers.nconvlayer as nconv
import lib.tfnconv.CustomLayers.downsamplelayer as down
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, UpSampling2D, Concatenate, Conv2D, Lambda, LeakyReLU, BatchNormalization, Add

#import model 
#path = '{0}\{1}'.format(pathlib.Path(__file__).parent.parent.absolute(), 'lib\pytracking')
#sys.path.append(path)
#tracker_module = importlib.import_module('pytracking.tracker.eco')

def create_model(padding, height, width, name):
    downs = 4
    filters = 2
    ks1 = 5
    ks2 = 3
    upsampling = UpSampling2D(size = 2, interpolation='nearest', name='up')
    conc = Concatenate(axis=-1)

    x = Input((height, width, 2), name='z')
    c= Input((height, width, 1), name='c')

    nconv1 = nconv.Nconv(filters=filters, kernel_size=(ks1,ks1), padding=padding, name='nconv_1')           
    nconv2 = nconv.Nconv(filters=filters, kernel_size=(ks1,ks1), padding=padding, name='nconv_2')        
    nconv3 = nconv.Nconv(filters=filters, kernel_size=(ks1,ks1), padding=padding, name='nconv_3')

    nconv4 = nconv.Nconv(filters=filters, kernel_size=(ks2,ks2), padding=padding, name='nconv_4')
    nconv5 = nconv.Nconv(filters=filters, kernel_size=(ks2,ks2), padding=padding, name='nconv_5')
                
    nconv6 = nconv.Nconv(filters=filters,kernel_size=(ks2,ks2), padding=padding, name='nconv_6')            
    nconv7 = nconv.Nconv(filters=1, kernel_size=(1, 1), padding=padding, name='nconv_7')               
    
    X_OUT = []      
    C_OUT = []

    for j in range(x.shape[-1]):
        x1, c1 = nconv1([x[...,j][...,None], c])                
        x1, c1 = nconv2([x1, c1])                
        x1, c1 = nconv3([x1, c1])
                
        X1_ds = [x1]
        C1_ds = [c1]
        
        for i in range(downs):
            x1_dss, c1_dss = down.Downsample(X1_ds[i].shape, 'down{0}_{1}'.format(i+2, j))([X1_ds[i], C1_ds[i]])     
            x1_ds, c1_ds =  nconv2([x1_dss, c1_dss])
            x1_ds, c1_ds =  nconv3([x1_ds, c1_ds])
            X1_ds.append(x1_ds)            
            C1_ds.append(c1_ds)            
        
        first = True
        for i in range(downs, 0,-1):  
            if first:
                x1_up = upsampling(X1_ds[i])                
                c1_up = upsampling(C1_ds[i])                
                first = False
            else:
                x1_up = upsampling(x1_ds)                
                c1_up = upsampling(c1_ds)                
                    
            conc_x1 = conc([X1_ds[i-1] ,x1_up])            
            conc_c1 = conc([C1_ds[i-1] ,c1_up])     
            
            x1_ds, c1_ds = nconv4([conc_x1, conc_c1])            
            x1_ds, c1_ds = nconv5([x1_ds, c1_ds])
    
                
        x1, c1 = nconv6([x1_ds, c1_ds])
        xout1, cout1 = nconv7([x1, c1])            
        X_OUT.append(xout1)
        C_OUT.append(cout1)
        
    out = Concatenate(axis=-1)([x for x in X_OUT])
    cout = sum(C_OUT)
    out = Concatenate(axis=-1)([out, cout])    
    out = Lambda(lambda x:x, name = "est_out")(out)        
    return Model(inputs= [x, c], outputs=out, name = name)