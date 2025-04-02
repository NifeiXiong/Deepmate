import numpy as np
import math
# from .basetrack import BaseTrack, TrackState
# from .utils import matching
# from .utils.kalman_filter import KalmanFilterXYAH
# def on_predict_start():

def find_possibleimg(box,cls):
    if cls != 2:
        return 1
        print('交配')
    else:
        areas=[]
        Center_distances=[]
        diagonal = set()
        i=0
        for i in range (cls) :
            area = box[i][2]*box[i][3]
            areas.append(area)
            Center_distance = math.sqrt((box[i][2])**2+(box[i][3])**2)
            Center_distances.append(Center_distance)
            if i<cls-1:
                j=i+1
                x=0
                for j in range(cls):
                    x=math.sqrt((box[i][0]-box[j][0])**2+(box[i][1]-box[j][1])**2)
                    diagonal.add(x)
        if (min(areas)<0.5*(max(areas)) or min(diagonal)<0.6*(max(Center_distances))):
            print("交配")
            return 1
        else:
            print("非交配")
            return 0




# box=([[191, 149, 95, 86],
#         [167,  93,  64,  64]])
# # box= np.array(box)
# cls= 2
#
# find_possibleimg(box,cls)