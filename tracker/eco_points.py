from .eco_parameters import eco_parameters
import pathlib
import importlib
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import time
import sys

class FlowPoints():
    def init(self, p, image, params, tracker_module):
        start_time = time.time()
        tracker = tracker_module.get_tracker_class()(params)
        tracker.initialize(image, p)    
        out =  {'target_bbox': p['init_bbox'],
                'pos': p['pos'],
                'time': time.time() - start_time, 
                'score': 1.0 }
        return tracker, [out]
    
    def _track(self, tracker, image):
        t = tracker[0]
        outs = tracker[1]
        start_time = time.time()
        out = t.track(image)
        out['time'] = time.time() - start_time 
        out['pos'] =  t.pos[[1,0]].tolist()
        out['score'] = t.debug_info['max_score']
        outs.append(out)

    def __init__(self, p0, image, plot=False):
        self.shape = image.shape
        path = '{0}\{1}'.format(pathlib.Path(__file__).parent.parent.absolute(), 'lib\pytracking')
        sys.path.append(path)
        tracker_module = importlib.import_module('pytracking.tracker.eco')
        init_pos = []
               
        params = eco_parameters()
        bs = params.box_size
        for p in p0:
            px = p[0,0] - bs/2
            py = p[0,1] - bs/2
            init_pos.append({'init_bbox': [px,py,bs,bs], 'pos':[p[0,0], p[0,1]]}) 
        start_time = time.time()
        self.trackers = []
        i = 0
        while i < len(init_pos):        
            p = init_pos[i]
            t = self.init(p, image, params, tracker_module)
            self.trackers.append(t)
            i += 1
            if i % 10 == 0:
                print("{0}/{1} points initialized".format(i, len(init_pos)))
        
        print("Time: {0}".format(time.time() - start_time))
        if plot:
            plt.imshow(image)
            [plt.plot(t[1][0]['pos'][0], t[1][0]['pos'][1], 'ro', markersize=4) for t in self.trackers]            
            plt.show()
        
    def track(self,image, plot=False):
        start_time = time.time()
        [self._track(t, image) for t in self.trackers]
        print("Time: {0}".format(time.time() - start_time))
        if plot:
            diffx = [t[1][-1]['pos'][0] - t[1][0]['pos'][0] for t in self.trackers]
            diffy = [t[1][-1]['pos'][1] - t[1][0]['pos'][1] for t in self.trackers]
            posx = [t[1][0]['pos'][0] for t in self.trackers]
            posy = [t[1][0]['pos'][1] for t in self.trackers]
            plt.imshow(image)
            [plt.quiver(posx[i], posy[i], diffx[i], diffy[i], units='dots', scale_units ='xy', angles='xy', scale=1, color ='r')
             for i in range(len(self.trackers))]
            [plt.plot(t[1][0]['pos'][0], t[1][0]['pos'][1], 'ro', markersize=4) for t in self.trackers]
            [plt.plot(t[1][-1]['pos'][0], t[1][-1]['pos'][1], 'bo', markersize=4) for t in self.trackers]
            plt.show()
    
    def _get_points(self, org_idx, new_idx):
        #org_points = []
        new_points = []
        #flow_points = []
        S = []
        for idx in range(len(self.trackers)):
            tracker = self.trackers[idx]
            #org_points.append(tracker[-1][org_idx]['pos']) 
            new_points.append(tracker[-1][new_idx]['pos'])
            #flow_points.append([tracker[-1][new_idx]['pos'][0] - tracker[-1][org_idx]['pos'][0], 
            #                    tracker[-1][new_idx]['pos'][1] - tracker[-1][org_idx]['pos'][1]])
            scores = np.asarray([tracker[-1][i]['score'] for i in range(org_idx, new_idx + 1)])
            S.append(scores)            
        return np.asarray(np.round(new_points), dtype=np.int), np.asarray(S)