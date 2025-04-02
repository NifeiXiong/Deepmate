# Ultralytics YOLO 🚀, AGPL-3.0 license

from functools import partial
from ultralytics import YOLO
import numpy as np
import math
import torch
import gol
import cv2
import scipy.misc
from PIL import Image
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_yaml

from .bot_sort import BOTSORT
from .byte_tracker import BYTETracker

gol._init()
TRACKER_MAP = {'bytetrack': BYTETracker, 'botsort': BOTSORT}
x=1
n=0
distance = []
i = 0
last = 0
head = []
def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.

    Raises:
        AssertionError: If the tracker_type is not 'bytetrack' or 'botsort'.
    """
    if hasattr(predictor, 'trackers') and persist:
        return
    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**yaml_load(tracker))
    assert cfg.tracker_type in ['bytetrack', 'botsort'], \
        f"Only support 'bytetrack' and 'botsort' for now, but got '{cfg.tracker_type}'"
    trackers = []
    for _ in range(predictor.dataset.bs):
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
        trackers.append(tracker)
    predictor.trackers = trackers


def on_predict_postprocess_end(predictor):
    global n
    global x
    # im_array = r.plot()
    # im = Image.fromarray(im_array[..., ::-1])
    """Postprocess detected boxes and update with object tracking."""
    bs = predictor.dataset.bs
    im0s = predictor.batch[1]
    button = 0
    for i in range(bs):
        det = predictor.results[i].boxes.cpu().numpy()
        aaa = predictor.results[i].orig_img
        # file_name = r"/home/zhaona/XNF/fin_savedimg/1\test" + "时间（秒）" + str(i) + ".jpg"
        # cv2.imwrite(file_name, aaa)
        n=n+1

        if n==6:
            x += 1
            num = det.cls.size
            # print(num)
            box = det.xywh
            img_select = find_possibleimg(box,num)#寻找靠的近的帧
            n=0
            if img_select == 1:
                xy = det.xyxy
                
                xx=[]
                y=[]
                centre = [0,0]
                for j in range(num):             
                    xx.append(xy[j][0])
                    xx.append(xy[j][2])
                    y.append(xy[j][1])
                    y.append(xy[j][3])
                cropped = aaa[int(min(y))-10:int(max(y))+10, int(min(xx))-10:int(max(xx))+10]#裁剪图片定位到小鼠上
                if cropped.size !=  0:
                    centre_x = 0.5*(min(xx)+max(xx))
                    centre_y = 0.5*(min(y)+max(y))
                    centre[0] = centre_x
                    centre[1] = centre_y
                    find_mating(centre, cropped)#使用分类，寻找交配的帧
#                file_name = r"/home/zhaona/XNF/img_near/img4" + "时间（秒）" + str(x) + ".jpg"
#                if cropped.size !=  0:
#                    cv2.imwrite(file_name, cropped)
#                    print('saved')

        if len(det) == 0:
            continue
        tracks = predictor.trackers[i].update(det, im0s[i])
        if len(tracks) == 0:
            continue
        idx = tracks[:, -1].astype(int)
        predictor.results[i] = predictor.results[i][idx]
        predictor.results[i].update(boxes=torch.as_tensor(tracks[:, :-1]))



def register_tracker(model, persist):
    """
    Register tracking callbacks to the model for object tracking during prediction.

    Args:
        model (object): The model object to register tracking callbacks for.
        persist (bool): Whether to persist the trackers if they already exist.
    """
    model.add_callback('on_predict_start', partial(on_predict_start, persist=persist))
    model.add_callback('on_predict_postprocess_end', on_predict_postprocess_end)


def find_possibleimg(box,cls):
    if cls == 1:
        return 1
    elif cls > 2:
        return 0
    elif cls < 0:
        return 0
    
    elif cls == 2:
        Center_distances=[]
        diagonal = set()
        i=0
        for i in range (cls) :
            Center_distance = math.sqrt((box[i][2])**2+(box[i][3])**2)
            Center_distances.append(Center_distance)
            if i<cls-1:
                j=i+1
                x=0
                for j in range(cls):
                    x=math.sqrt((box[i][0]-box[j][0])**2+(box[i][1]-box[j][1])**2)
                    diagonal.add(x)
        diagonal.discard(0)
        if min(diagonal)<0.35*(max(Center_distances)):
            return 1
        else:
            return 0

def find_mating(centre,cropped):#寻找可能在交配的帧
    model = YOLO(r"weight/class.pt")
    results = model(cropped)
    probs = results[0].probs.data.tolist()
    a = np.argmax(probs)
#    print(np.argmax(probs))
    if a == 0:
        second = x%60
        minute = (x//60)%60
        hour = (x//60)//60
        word =  str(hour) + "h" + str(minute) + "min" + str(second)+ "s"
#        file_name = r"/home/zhaona/XNF/fin_savedimg/1/time_"+ str(hour) + "h" + str(minute) + "min" + str(second)+ "s"   + ".jpg"
#        if cropped.size !=  0:
#            cv2.imwrite(file_name, cropped)
        saved_time(centre, word)
    else:
        print('no')


def saved_time(centre, word):#寻找交配时间段
    centres = []
    global i
    global last
    global head
    global distance#记录每组交配段中心点
    centres.append(centre)
    # print('aaaaaaaaaaaaaaaaa')
    print('Mating...')
    time = []
    time_s = []
    if i >0:
        time = gol.get_value("time")
        time_s = gol.get_value("time_s")
    now = x
    gap = now - last
    if gap > 10:#如果间隔大于10秒
        dis_interval = []#计算位移的间隔
            
        if len(head) == 0:
            head.append(now)
            gol.set_value('head',head)
        else:
            head = gol.get_value('head')
            head.append(now)
            gol.set_value('head',head)
            if last-head[-2]<3:
                time.pop()
                time.pop()
                time_s.pop()
                time_s.pop()
                i = i - 1
        if i > 0:
            if time[-1] == 0:
                time.pop()
                time.pop()
                time_s.pop()
                time_s.pop()
                i = i-1
        i += 1
        time.append('第'+ "{:03}".format(i)+'段：'+word)
        time.append(0)
        time_s.append('第'+ "{:03}".format(i)+'段:(s)'+str(x))
        time_s.append(0)
    else:
        time[-1] = word
        time_s[-1] = str(x)
    last = now
    gol.set_value("time",time)
    gol.set_value("time_s",time_s)

# def find_possibleimg(box, cls):
#     if cls == 1:
#         return 1
#     else:
#         areas = []
#         Center_distances = []
#         diagonal = set()
#         i = 0
#         for i in range(cls):
#             area = box[i][2] * box[i][3]
#             areas.append(area)
#             Center_distance = math.sqrt((box[i][2]) ** 2 + (box[i][3]) ** 2)
#             Center_distances.append(Center_distance)
#             if i < cls - 1:
#                 j = i + 1
#                 x = 0
#                 for j in range(cls):
#                     x = math.sqrt((box[i][0] - box[j][0]) ** 2 + (box[i][1] - box[j][1]) ** 2)
#                     diagonal.add(x)
#         diagonal.discard(0)
#         if (min(areas) < 0.4 * (max(areas)) or min(diagonal) < 0.4 * (max(Center_distances))):
#             print("交配")
#             return 1
#         else:
#             print("非交配")
#             return 0
