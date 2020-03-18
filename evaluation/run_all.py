from evaluation.run_patient import run_patient
from utils.read_data import ImageReader
import pickle

def run_all():
    data_path = './data/MRI_Seg/'
    seg_path = './data/MRI_Seg/Seg'
    img_heigth = 256
    img_width = 256
    img_reader = ImageReader(data_path, seg_path, img_heigth, img_width)
    model = create_model('SAME', 256, 256, 'final_model')
    model.load_weights('./interpolation/saved_model/final_model.h5')
    methods = {
        'eco':{'points': 150,
                  'nconv' : {'model':model},
                  'tps': True
                 },
        'bspline':{'spacing':20.0,
                      'metric':'mse'
                     },
        'demons':{'intDiffThresholds':10}
    }
    
    all_data = []
    for patient in range(0,32):
        result = run_patient(img_reader, patient, 4, methods, 0, 20)
        data = {'patient': patient,
                'data':result}
        all_data.append(data)
    
    with open('./all_result_{0}.pickle'.format(z_pos), 'wb') as handle:
        pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__": 
    run_all() 

