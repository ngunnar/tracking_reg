from evaluation.run_patient_camus import run_patient
from utils.camus_reader import camus_reader
import pickle
import glob

def run_all():    
    pat_dir = glob.glob('/data/Flow/camus/training/*')
    pat_dir.sort()
    
    img_heigth = 256
    img_width = 256
    
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
    for pd in pat_dir:
        patient = pd.split('/')[-1]
        p_2ch_i = glob.glob('{0}/*2CH_sequence.mhd'.format(pd))
        p_2ch_ed_gt = glob.glob('{0}/*2CH_ED_gt.mhd'.format(pd))
        p_2ch_es_gt = glob.glob('{0}/*2CH_ES_gt.mhd'.format(pd))
    
        p_4ch_i = glob.glob('{0}/*4CH_sequence.mhd'.format(pd))
        p_4ch_ed_gt = glob.glob('{0}/*4CH_ED_gt.mhd'.format(pd))
        p_4ch_es_gt = glob.glob('{0}/*4CH_ES_gt.mhd'.format(pd))
    
        if len(p_2ch_i) == 0:
            continue
        
        assert len(p_2ch_i) == 1
        assert len(p_2ch_es_gt) == 1
        assert len(p_2ch_ed_gt) == 1
        assert len(p_4ch_i) == 1
        assert len(p_4ch_es_gt) == 1
        assert len(p_4ch_ed_gt) == 1
    
        i, seg_s, seg_e = camus_reader(p_2ch_i[0], p_2ch_ed_gt[0], p_2ch_es_gt[0], 3, (img_heigth, img_width))
        
        result = run_patient(i, seg_s, seg_e, methods)
        data = {'patient': patient,
                'type': '2CH',
                'data':result}
        
        all_data.append(data)
        
        i, seg_s, seg_e = camus_reader(p_4ch_i[0], p_4ch_ed_gt[0], p_4ch_es_gt[0], 3, (img_heigth, img_width))
        result = run_patient(i, seg_s, seg_e, methods)
        data = {'patient': patient,
                'type': '4CH',
                'data':result}
        all_data.append(data)
    
    
    with open('./all_result_CAMUS.pickle', 'wb') as handle:
        pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return all_data
if __name__ == "__main__": 
    run_all() 

