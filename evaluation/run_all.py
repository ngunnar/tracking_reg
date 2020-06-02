from evaluation.run_patient import run_patient
from utils.camus_reader import ImageReader
import pickle

def run_all(suffix=None):
    data_path = '/data/MRI_Seg/'
    seg_path = '/data/MRI_Seg/Seg'
    img_heigth = 256
    img_width = 256
    z_pos = 4
    img_reader = ImageReader(data_path, seg_path, img_heigth, img_width)
    methods = {
        'eco':{'points': 150,
               'nconv' : {'model':'/home/ngune07417/project/tracking_reg/interpolation/saved_model/final_model.h5'},
                'tps': True
                 },
        'bspline':{'spacing':20.0,
                   'metric':'mse'
                   },
        'demons':{'intDiffThresholds':10}
    }
    
    all_data = []
    pat_tot = 33
    for patient in range(0,pat_tot):
        print("Running patient {0}".format(patient))
        result = run_patient(img_reader, patient, z_pos, methods, 0, 20)
        data = {'patient': patient,
                'data':result}
        all_data.append(data)
    
    with open('./all_result_{0}_{1}.pickle'.format(z_pos, suffix), 'wb') as handle:
        pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return all_data
if __name__ == "__main__": 
    run_all() 

