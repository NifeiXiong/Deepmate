import pandas as pd
import gol


# 读取Excel文件
file_path = r"USV_data/output/output.csv" # 替换为你的Excel文件路径

df = pd.read_csv(file_path)

# 初始化总时长
total_duration = 0

# 找到所有pre_Results为1的行
in_sequence = False
start_time = 0

for index, row in df.iterrows():
    if row['pre_Results'] == 1:
        if not in_sequence:  # 如果是新的一段
            start_time = row['Begin Time (s)']
            in_sequence = True
        end_time = row['End Time (s)']
    else:
        if in_sequence:  # 如果之前在一段连续的1中，现在遇到0，计算时长
            duration = end_time - start_time
            if duration > 0:  # 只累加大于0的时长
                total_duration += duration
            in_sequence = False

# 如果最后一行还是连续1的结束，则再加上最后一段的时长
if in_sequence:
    duration = end_time - start_time
    if duration > 0:  # 只累加大于0的时长
        total_duration += duration

# 打印结果
print(f"Total duration with pre_Results equal to 1 (excluding negative durations): {total_duration} seconds")
