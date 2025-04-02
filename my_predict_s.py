import sys
import os
from ultralytics import YOLO
import gol
from moviepy.video.io.VideoFileClip import VideoFileClip



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def mat_behavior_predict(video_address, output_address):

    gol._init()
    #输入第17行，输出第29和45行

    print(1)
    #gol.set_value("button", 0)
    # Load an official or custom model
    model = YOLO(r'weight/object.pt')  # Load an official Detect model#我固定的

    # Perform tracking with the model
    #results = model.track(r"D:\Mouse_sound\others\vidoes\2023 10 14.avi", show=True)  # Tracking with default tracker
    results = model.track(video_address,show=True,  tracker="bytetrack.yaml")  # Tracking with ByteTrack tracker
    time = gol.get_value("time")
    #with open("/home/zhaona/XNF/fin_savedimg/1.txt", "w") as f:
    #    f.write(str(time))
    if time[-1] == 0:
        time.pop()
        time.pop()
    fin_time = []
    i=0
    while i <len(time):
        fin_time.append(time[i]+'--'+time[i+1])
        i+= 2
    Output_minute_address = output_address.replace("\\Output_s.txt", "") + "/Output_minute.txt"
    with open(Output_minute_address, "w") as f:
        f.write(str(fin_time))

    #按秒保存
    time_s = gol.get_value("time_s")
    #with open("/home/zhaona/XNF/fin_savedimg/1.txt", "w") as f:
    #    f.write(str(time))
    if time_s[-1] == 0:
        time_s.pop()
        time_s.pop()
    fin_time_s = []
    i=0
    while i <len(time_s):
        fin_time_s.append(time_s[i]+'--'+time_s[i+1])
        i+= 2

    file_write_obj = open(output_address, 'w')   # 新文件
    for var in fin_time_s:
        file_write_obj.write(var)   # 逐行写入
        file_write_obj.write('\n')
    file_write_obj.close()

    extract_clips_from_file(video_address, output_address)

def extract_clips_from_file(video_file, output_address):
    # 读取文本文件
    output_dir = output_address.replace("\\Output_s.txt", "")
    try:
        with open(output_address, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except UnicodeDecodeError:
        with open(output_address, 'r', encoding='gbk') as file:
            lines = file.readlines()

    # 遍历每一行
    for line in lines:
        # 找到行中的时间段
        parts = line.strip().split(':')
        segment_name = parts[0]
        time_range = parts[1].strip().strip('(s)').split('--')

        start_time = int(time_range[0])
        end_time = int(time_range[1])

        # 提取视频片段
        with VideoFileClip(video_file) as video:
            # 截取指定时间段
            clip = video.subclip(start_time, end_time)
            # 生成输出文件路径
            output_filename = os.path.join(output_dir, f"{segment_name}.mp4")
            # 保存视频片段
            clip.write_videofile(output_filename, codec='libx264')
            print(f"Saved: {output_filename}")

if __name__ == "__main__":
    video_address = r"../../video_data/input/test_mating.mp4"
    output_address = "../../video_data/output/Output_s.txt"
    mat_behavior_predict(video_address, output_address)


