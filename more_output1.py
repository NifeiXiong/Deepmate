import pandas as pd

# 定义一个函数来读取 CSV 文件并生成重复值的列表，并将结果保存到现有的 Excel 文件中
def generate_repeated_values(input_file, output_file):
    # 读取 CSV 文件
    df = pd.read_csv(input_file)

    # 初始化一个空列表来存储重复的值
    repeated_values = []

    # 遍历数据框中的每一行
    for count, value in zip(df['USVs_per_segment'], df['predicted_results']):
        # 确保 count 是整数
        count = int(count)

        # 将 'predicted_results' 列中的值按 'USVs_per_segment' 的次数进行重复
        repeated_values.extend([value] * count)

    # 创建一个新的数据框来保存结果
    result_df = pd.DataFrame({'Repeated_Results': repeated_values})

    # 读取现有的 Excel 文件
    existing_data = pd.read_excel(output_file, sheet_name='Sheet1')

    # 计算现有数据和新数据的长度差
    len_diff = len(existing_data) - len(result_df)

    # 如果新数据的行数少于现有数据，补齐 0
    if len_diff > 0:
        # 在 result_df 末尾补 0
        padding = pd.DataFrame({'Repeated_Results': [0] * len_diff})
        result_df = pd.concat([result_df, padding], ignore_index=True)
    # 如果新数据的行数多于现有数据，在现有数据中补齐 0
    elif len_diff < 0:
        padding = pd.DataFrame({col: [0] * abs(len_diff) for col in existing_data.columns})
        existing_data = pd.concat([existing_data, padding], ignore_index=True)

    # 将新数据作为新列添加到现有数据框中
    existing_data['pre_Results'] = result_df['Repeated_Results']

    # 将更新后的数据写回 Excel 文件
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        existing_data.to_excel(writer, index=False, sheet_name='Sheet1')

with open("address.txt", "r", encoding="utf-8") as file:
    text = file.read()
    print(text)
# 指定输入文件路径和输出 Excel 文件路径
input_file_path = r"USV_data/output/output.csv"  # 输入文件路径
output_file_path = text # 输出 Excel 文件路径

# 生成重复的值并保存输出
generate_repeated_values(input_file_path, output_file_path)

import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件并选择相关列
file_path = text # 替换为你的Excel文件路径
df = pd.read_excel(file_path)

# 选择需要分析的特征列
features = [
    'Call Length (s)', 'Principal Frequency (kHz)', 'Low Freq (kHz)', 'High Freq (kHz)',
    'Delta Freq (kHz)', 'Frequency Standard Deviation (kHz)', 'Slope (kHz/s)',
    'Sinuosity', 'Mean Power (dB/Hz)', 'Tonality', 'Peak Freq (kHz)', 'interval'
]

# 根据 'pre_Results' 列划分数据
class_1 = df[df['pre_Results'] == 1]
class_0 = df[df['pre_Results'] == 0]

# 计算每个特征列在两类中的平均值
mean_class_1 = class_1[features].mean()
mean_class_0 = class_0[features].mean()

# 计算pre_Results中 0 和 1 的数量
count_0 = len(class_0)
count_1 = len(class_1)

# 创建一个新的DataFrame来存储平均值对比
mean_comparison = pd.DataFrame({
    'Class 0 Mean': mean_class_0,
    'Class 1 Mean': mean_class_1
})

# 绘制平均值对比图
plt.figure(figsize=(14, 7))
mean_comparison.plot(kind='bar', width=0.8)
plt.title('Feature Mean Comparison Between Class 0 and Class 1')
plt.xlabel('Features')
plt.ylabel('Mean Values')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()

# 显示平均值对比图
plt.show()

# 绘制数量对比图
plt.figure(figsize=(6, 6))
plt.bar(['Class 0', 'Class 1'], [count_0, count_1], color=['blue', 'red'])
plt.title('Count of Class 0 and Class 1')
plt.ylabel('Count')
plt.grid(axis='y')

# 显示数量对比图
plt.show()


