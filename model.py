from tensorflow.keras import Model
from tensorflow.keras.layers import Input, UpSampling2D, Concatenate
from CustomLayers.nconvlayer import Nconv
from CustomLayers.downsamplelayer import Downsample

def create_model(padding, height, width, name):
        filters = 2
        kernel_size = (5,5)
        nconv1 = Nconv(filters=filters,kernel_size=kernel_size, padding=padding, name='nconv1')
        nconv2 = Nconv(filters=filters,kernel_size=kernel_size, padding=padding, name='nconv2')
        nconv3 = Nconv(filters=filters,kernel_size=kernel_size, padding=padding, name='nconv3')

        upsampling = UpSampling2D(size = 2, interpolation='nearest', name='up')
        
        kernel_size = (3,3)
        filters = filters*2
        nconv4 = Nconv(filters=filters,kernel_size=kernel_size, padding=padding, name='nconv4')
        nconv5 = Nconv(filters=filters,kernel_size=kernel_size, padding=padding, name='nconv5')
        nconv6 = Nconv(filters=filters,kernel_size=kernel_size, padding=padding, name='nconv6')
        
        kernel_size = (1,1)
        filters = 1
        nconv7 = Nconv(filters=filters,kernel_size=(1,1), padding=padding, name='nconv7')

        inputs1 = Input((height, width, 1))
        inputs2 = Input((height, width, 1))

        x1, c1 = nconv1([inputs1, inputs2])
        x1, c1 = nconv2([x1, c1])
        x1, c1 = nconv3([x1, c1])

        x1_ds, c1_ds = Downsample(x1.shape, 'down1')([x1,c1])
        x2_ds, c2_ds = nconv2([x1_ds, c1_ds])
        x2_ds, c2_ds = nconv3([x2_ds, c2_ds])
        
        x2_dss, c2_dss = Downsample(x2_ds.shape, 'down2')([x2_ds,c2_ds])
        x3_ds, c3_ds  = nconv2([x2_dss, c2_dss])

        x3_dss, c3_dss = Downsample(x3_ds.shape, 'down3')([x3_ds,c3_ds])
        x4_ds, c4_ds  = nconv2([x3_dss, c3_dss])
        
        x4 = upsampling(x4_ds)
        c4 = upsampling(c4_ds)
        conc_x34 = Concatenate(axis=-1)([x3_ds,x4])
        conc_c34 = Concatenate(axis=-1)([c3_ds,c4])
        x34_ds, c34_ds = nconv4([conc_x34, conc_c34])

        x34 = upsampling(x34_ds)
        c34 = upsampling(c34_ds)
        conc_x23 = Concatenate(axis=-1)([x2_ds,x34])
        conc_c23 = Concatenate(axis=-1)([c2_ds,c34])
        x23_ds, c23_ds = nconv5([conc_x23, conc_c23])
        
        x23 = upsampling(x23_ds)
        c23 = upsampling(c23_ds)
        conc_x12 = Concatenate(axis=-1)([x23, x1])
        conc_c12 = Concatenate(axis=-1)([c23, c1])
        xout, cout = nconv6([conc_x12, conc_c12])

        xout, cout = nconv7([xout, cout])
        return Model(inputs= [inputs1, inputs2], outputs=Concatenate(axis=-1)([xout, cout]), name = name)


if __name__ == "__main__":    
    model = create_model(padding="SAME", height = 512, width=512, name="NCONV_NET")
    print(model.summary())