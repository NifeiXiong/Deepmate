import cv2
import numpy as np
import math
from tqdm import tqdm
import tkinter.filedialog


def video_cut(input_address, output_address):
    # 逆时针旋转
    # 输入选择文件，输出第92行
    def Nrotate(angle, valuex, valuey, pointx, pointy):
        angle = (angle / 180) * math.pi
        valuex = np.array(valuex)
        valuey = np.array(valuey)
        nRotatex = (valuex - pointx) * math.cos(angle) - (valuey - pointy) * math.sin(angle) + pointx
        nRotatey = (valuex - pointx) * math.sin(angle) + (valuey - pointy) * math.cos(angle) + pointy
        return (nRotatex, nRotatey)

    # 顺时针旋转
    def Srotate(angle, valuex, valuey, pointx, pointy):
        angle = (angle / 180) * math.pi
        valuex = np.array(valuex)
        valuey = np.array(valuey)
        sRotatex = (valuex - pointx) * math.cos(angle) + (valuey - pointy) * math.sin(angle) + pointx
        sRotatey = (valuey - pointy) * math.cos(angle) - (valuex - pointx) * math.sin(angle) + pointy
        return (sRotatex, sRotatey)

    # 将四个点做映射
    def rotatecordiate(angle, rectboxs, pointx, pointy):
        output = []
        for rectbox in rectboxs:
            if angle > 0:
                output.append(Srotate(angle, rectbox[0], rectbox[1], pointx, pointy))
            else:
                output.append(Nrotate(-angle, rectbox[0], rectbox[1], pointx, pointy))
        return output

    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            a.append(x)
            b.append(y)
            cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 0), thickness=1)
            cv2.imshow("image", img)

    def imagecrop(image, box):
        xs = [x[1] for x in box]
        ys = [x[0] for x in box]

        cropimage = image[min(xs):max(xs), min(ys):max(ys)]
        return cropimage

    filename = input_address
    cap = cv2.VideoCapture(filename)
    frame_width = int(cap.get(3))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    print(FPS)
    # print(frames)
    frame_height = int(cap.get(4))
    frame_FPS = int(cap.get(5))
    ret, frame = cap.read()
    img = frame
    a = []
    b = []

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img)
    cv2.waitKey(0)

    cnt = np.array([[a[0], b[0]], [a[1], b[1]], [a[2], b[2]], [a[3], b[3]]])  # 必须是array数组的形式
    rect = cv2.minAreaRect(cnt)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    box_origin = cv2.boxPoints(rect)  # box_origin为[(x0,y0),(x1,y1),(x2,y2),(x3,y3)]
    box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)

    M = cv2.getRotationMatrix2D(rect[0], rect[2], 1)
    dst = cv2.warpAffine(img, M, (2 * img.shape[0], 2 * img.shape[1]))
    box = np.int0(box)

    box = rotatecordiate(rect[2], box_origin, rect[0][0], rect[0][1])

    final_image = imagecrop(dst, np.int0(box))
    cv2.namedWindow("cut_image")
    cv2.imshow("cut_image", final_image)
    cv2.waitKey(0)

    frame_FPS = cap.get(cv2.CAP_PROP_FPS)
    print(frame_FPS)
    out = cv2.VideoWriter(output_address, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), FPS,
                          (final_image.shape[1], final_image.shape[0]))
    timeC = 0
    for i in tqdm(range(frames)):
        ret, frame = cap.read()
        if ret:
            # timeC = timeC + 1
            # 每隔 10 帧进行操作
            # if (timeC % 3 != 0):
            #       ret = cap.grab()
            #      continue
            # img = frame
            M = cv2.getRotationMatrix2D(rect[0], rect[2], 1)
            dst = cv2.warpAffine(img, M, (2 * img.shape[0], 2 * img.shape[1]))
            box = np.int0(box)

            box = rotatecordiate(rect[2], box_origin, rect[0][0], rect[0][1])

            final_image = imagecrop(dst, np.int0(box))
            # cv2.imshow('frame', final_image)
            out.write(final_image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    # 6.释放视频对象
    cap.release()
    out.release()

if __name__ == "__main__":
    input_address = r"D:\Projects\all_model\video_model\video_data\input\test_clip.mp4"
    output_address = r"'outpy.avi'"
    video_cut(input_address, output_address)
