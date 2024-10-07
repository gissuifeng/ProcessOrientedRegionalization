import arcpy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import heapq 
from typing import Dict, List, Tuple
import random
import numpy as np
import math
import bisect
from scipy import signal
import copy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# 指定字体，确保系统中有这个字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

#### Tool functions Start ####

# 读取纵向邻居信息
def create_lengthwise_neighbor_dict(file_path):


    # 初始化一个字典来存储邻居信息
    neighbor_dict = {}

    try:
        # 尝试打开并读取文件
        with open(file_path, 'r') as file:
            # 逐行读取文件内容
            for line in file:
                # 去除行首尾的空白字符，并按空格或制表符分割
                # 这里假设ID之间用空格或制表符分隔，如果使用其他分隔符，请相应修改
                regions = line.strip().split()

                # 如果分割后不是两个元素，则跳过此行
                if len(regions) != 2:
                    arcpy.AddMessage(f"警告: 邻居文件跳过格式不正确的行: {line.strip()}")
                    continue

                # 获取两个相邻的区域ID
                region1, region2 = regions

                # 将region2添加到region1的邻居列表中
                if region1 in neighbor_dict:
                    if region2 not in neighbor_dict[region1]:
                        neighbor_dict[region1].append(region2)
                else:
                    neighbor_dict[region1] = [region2]

                # 将region1添加到region2的邻居列表中
                if region2 in neighbor_dict:
                    if region1 not in neighbor_dict[region2]:
                        neighbor_dict[region2].append(region1)
                else:
                    neighbor_dict[region2] = [region1]

    except FileNotFoundError:
        # 如果文件不存在，打印错误信息
        arcpy.AddMessage(f"错误：未找到文件 '{file_path}'。")
    except IOError:
        # 如果文件无法读取（比如权限问题），打印错误信息
        arcpy.AddMessage(f"错误：无法读取文件 '{file_path}'。")

    # 返回创建好的邻居字典
    return neighbor_dict


# 读取邻居信息
def create_crosswise_neighbor_dict(file_path):
    # 初始化一个空字典来存储邻居信息
    neighbor_dict = {}

    try:
        # 尝试打开并读取文件
        with open(file_path, 'r') as file:
            # 逐行读取文件内容
            for line in file:
                # 去除行首尾的空白字符，并按空格分割成ID列表
                ids = line.strip().split(",")

                # 如果分割后的列表为空（即空行），则跳过此行
                if len(ids) < 1:
                    continue

                # 列表的第一个元素是当前区域的ID
                current_region = ids[0]

                # 列表的剩余元素是邻居区域的ID
                neighbors = ids[1:]

                # 将当前区域及其邻居添加到字典中
                neighbor_dict[current_region] = neighbors

    except FileNotFoundError:
        # 如果文件不存在，打印错误信息
        arcpy.AddMessage(f"错误：未找到文件 '{file_path}'。")
    except IOError:
        # 如果文件无法读取（比如权限问题），打印错误信息
        arcpy.AddMessage(f"错误：无法读取文件 '{file_path}'。")

    # 返回创建好的邻居字典
    return neighbor_dict



# 函数使用说明：
# 1. 调用函数时，传入包含邻居信息的文件路径
# 2. 函数会返回一个字典，其中键是区域ID，值是该区域邻居ID的列表
# 3. 如果文件不存在或无法读取，函数会打印错误信息并返回一个空字典

# 注意事项：
# - 确保输入文件的每一行格式正确：第一个ID是当前区域，后面的ID是邻居
# - 函数假设ID之间用空格分隔，如果使用其他分隔符，需要修改 split() 方法
# - 函数会自动跳过空行，增加了对输入文件的容错性
# - 如果文件很大，可能需要考虑使用更高效的读取方式，如逐行读取而不是一次性读入内存


#时序列表创建函数


# 从给定的文件路径读取时序信息，创建并返回时序列表。
def create_lengthwise_time_series_list(file_path):
    """
    从给定的文件路径读取时序信息，创建并返回时序列表。

    参数:
    file_path (str): 包含时序信息的文件路径

    返回:
    list: 包含所有时序序列的列表，每个序列是一个列表的元组 (值, 时间字符串)
    """

    # 初始化一个字典来存储所有的时序序列
    series_dict = {}

    # 设置起始时间（用于两列数据的情况）
    start_time = datetime(2024, 1, 1, 0, 0)

    try:
        # 尝试打开并读取文件
        with open(file_path, 'r') as file:
            # 逐行读取文件内容
            for line in file:
                if line.startswith("id"):
                    continue
                # 去除行首尾的空白字符，并按逗号分割
                items = line.strip().split(',')

                # 如果分割后的列表少于2个或多于3个元素，则跳过此行
                if len(items) < 2 or len(items) > 3:
                    arcpy.AddMessage(f"警告: 时序文件跳过格式不正确的行: {line.strip()}")
                    continue

                # 提取序列编号和值
                sequence_number = items[0]
                value = float(items[1])  # 将值转换为小数

                # 处理时间信息
                if len(items) == 3:
                    # 如果是三列数据，直接使用提供的时间
                    time_str = items[2]
                else:
                    # 如果是两列数据，生成时间
                    if sequence_number not in series_dict:
                        # 如果是新序列，使用起始时间
                        current_time = start_time
                    else:
                        # 如果序列已存在，在最后一个时间基础上加5分钟
                        last_time = datetime.strptime(series_dict[sequence_number][-1][1], "%Y-%m-%d %H:%M")
                        current_time = last_time + timedelta(minutes=5)

                    time_str = current_time.strftime("%Y-%m-%d %H:%M")

                # 将 (值, 时间) 元组添加到对应的序列
                if sequence_number in series_dict:
                    series_dict[sequence_number].append((value, time_str))
                else:
                    series_dict[sequence_number] = [(value, time_str)]

    except FileNotFoundError:
        # 如果文件不存在，打印错误信息
        arcpy.AddMessage(f"错误：未找到文件 '{file_path}'。")
        return []
    except IOError:
        # 如果文件无法读取（比如权限问题），打印错误信息
        arcpy.AddMessage(f"错误：无法读取文件 '{file_path}'。")
        return []
    except ValueError as e:     #ValueError
        # 如果在转换过程中出现错误（例如，非整数值），打印错误信息
        arcpy.AddMessage(f"错误：处理数据时出现问题 - {str(e)}")
        return []

    # 将字典返回
    return series_dict


"""
# 使用示例：

nei_file_path = r'../dataset/处理好的数据/邻接矩阵最终版.csv'
neighbor_dict = create_crosswise_neighbor_dict(nei_file_path)
print(neighbor_dict)

# 使用示例：
time_series_filepath = r'../dataset/处理好的数据/valueDataIdSwapShort2.csv'
time_series_dict = create_lengthwise_time_series_list(time_series_filepath)
for i, series in time_series_dict.items():
    print(f"序列 {i}:")
    print(series)
    print()
# print(time_series_dict)
"""

#### Tool functions End ####


#### Value Similarity Start ####

def calculate_value_similarity(ts1, ts2) -> float:
    """计算两个时间序列的值相似性"""
    values1 = [v for v, _ in ts1]
    values2 = [v for v, _ in ts2]
    return np.mean(np.abs(np.array(values1) - np.array(values2)))

def compute_Edis_similarity(ts1, ts2):
    # 提取值序列
    values1 = [v for v, _ in ts1]
    values2 = [v for v, _ in ts2]

    # 确定较短序列的长度
    min_length = min(len(values1), len(values2))

    # 计算匹配部分的相似性
    matching_similarity = sum(abs(values1[i] - values2[i]) for i in range(min_length))

    # 计算平均相似性
    average_similarity = matching_similarity / min_length

    # 计算不匹配部分的长度
    unmatched_length = abs(len(values1) - len(values2))

    # 计算不匹配部分的相似性
    unmatched_similarity = average_similarity * unmatched_length

    # 计算总相似性
    total_similarity = matching_similarity + unmatched_similarity

    return total_similarity


def coumput_dtw_similarity(ts1, ts2):
    # 从时间序列中只提取值，忽略时间戳
    seq1 = [item[0] for item in ts1]
    seq2 = [item[0] for item in ts2]

    n, m = len(seq1), len(seq2)

    # 初始化成本矩阵
    # 第一行和第一列设置为无穷大，除了(0,0)位置
    cost_matrix = np.zeros((n + 1, m + 1))
    cost_matrix[0, 1:] = np.inf
    cost_matrix[1:, 0] = np.inf

    # 填充成本矩阵
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            try:
            # 计算当前两个点之间的距离
                cost = abs(seq1[i - 1] - seq2[j - 1])
            except Exception  as e:
                arcpy.AddMessage(e)
            # 选择最小成本路径
            cost_matrix[i, j] = cost + min(cost_matrix[i - 1, j],  # 插入
                                           cost_matrix[i, j - 1],  # 删除
                                           cost_matrix[i - 1, j - 1])  # 匹配

    # 返回右下角的值，即总的DTW距离
    return cost_matrix[n, m]

#### Value Similarity End ####

#### Sahpe Similarity Start ####

def parse_datetime(date_string):
    return datetime.strptime(date_string, "%Y-%m-%d %H:%M")

def find_time_intersection(ts1, ts2):
    start1, end1 = ts1[0][1], ts1[-1][1]
    start2, end2 = ts2[0][1], ts2[-1][1]
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    return intersection_start, intersection_end

def interpolate_value(t, t1, t2, v1, v2):
    if t1 == t2:
        return v1
    ratio = ((t - t1).total_seconds()/60) / ((t2 - t1).total_seconds()/60)
    return v1 + ratio * (v2 - v1)

def equalize_time_series(ts1, ts2):
    ts1 = [(v, parse_datetime(t)) for v, t in ts1]
    ts2 = [(v, parse_datetime(t)) for v, t in ts2]

    intersection_start, intersection_end = find_time_intersection(ts1, ts2)
    all_timestamps = sorted(set([t for _, t in ts1 + ts2 if intersection_start <= t <= intersection_end]))

    def interpolate_series(ts):
        result = []
        for t in all_timestamps:
            if t < ts[0][1] or t > ts[-1][1]:
                continue
            i = bisect.bisect_left([x[1] for x in ts], t)
            if i == 0 or ts[i][1] == t:
                result.append((ts[i][0], t))
            else:
                v = interpolate_value(t, ts[i - 1][1], ts[i][1], ts[i - 1][0], ts[i][0])
                result.append((v, t))
        return result

    ts1_eq = interpolate_series(ts1)
    ts2_eq = interpolate_series(ts2)

    min_len = min(len(ts1_eq), len(ts2_eq))
    ts1_eq = ts1_eq[:min_len]
    ts2_eq = ts2_eq[:min_len]

    ts1_eq = [(v, t.strftime("%Y-%m-%d %H:%M")) for v, t in ts1_eq]
    ts2_eq = [(v, t.strftime("%Y-%m-%d %H:%M")) for v, t in ts2_eq]

    return ts1_eq, ts2_eq

def calculate_slope(t1, t2, v1, v2):
    time_diff = (parse_datetime(t2) - parse_datetime(t1)).total_seconds()/60
    return (v2 - v1) / time_diff if time_diff != 0 else 0

def determine_pattern(k_current, k_next, epsilon):
    if (k_next > epsilon and k_current < epsilon) or ( k_next > epsilon and k_current > epsilon and k_next - k_current > 0):
        return 3  # 加速上升
    elif k_next > epsilon and k_current > epsilon and k_next - k_current == 0:
        return 2  # 水平上升
    elif k_next > epsilon and k_current > epsilon and k_next - k_current < 0:
        return 1  # 减速上升
    elif (k_next < -epsilon and k_current > -epsilon) or (k_next < -epsilon and k_current < -epsilon and k_next - k_current < 0):
        return -3  # 加速下降
    elif k_next < -epsilon and k_current < -epsilon and k_next - k_current == 0:
        return -2  # 水平下降
    elif k_next < -epsilon and k_current < -epsilon and k_next - k_current > 0:
        return -1  # 减速下降
    elif -epsilon < k_next < epsilon:
        return 0  # 不变
    else:
        return None  # 未定义模式

def pattern_analysis(time_series, epsilon=math.tan(math.pi / 360)):
    patterns = []
    time_lengths = []
    slopes = []

    for i in range(len(time_series) - 2):
        v1, t1 = time_series[i]
        v2, t2 = time_series[i + 1]
        v3, t3 = time_series[i + 2]

        k_current = calculate_slope(t1, t2, v1, v2)
        k_next = calculate_slope(t2, t3, v2, v3)

        pattern = determine_pattern(k_current, k_next, epsilon)
        if pattern is not None:
            patterns.append(pattern)
            time_lengths.append((parse_datetime(t2) - parse_datetime(t1)).total_seconds()/60)
            slopes.append(k_current)

    return patterns, time_lengths, slopes

def calculate_similarity(ts1, ts2):
    patterns1, time_lengths1, _ = pattern_analysis(ts1)
    patterns2, time_lengths2, _ = pattern_analysis(ts2)

    min_length = min(len(patterns1), len(patterns2))
    patterns1 = patterns1[:min_length]
    patterns2 = patterns2[:min_length]
    time_lengths1 = time_lengths1[:min_length]
    time_lengths2 = time_lengths2[:min_length]

    total_time = sum(time_lengths1)
    similarity = 0

    for i in range(min_length):
        t_wi = time_lengths1[i] / total_time
        pattern_diff = abs(patterns1[i] - patterns2[i])
        value_diff = abs((ts1[i + 2][0] - ts1[i + 1][0]) - (ts2[i + 2][0] - ts2[i + 1][0]))
        similarity += t_wi * pattern_diff * value_diff

    return similarity

def calculate_shape_similarity(TS_1, TS_2):
    # 等序列化
    TS_1_eq, TS_2_eq = equalize_time_series(TS_1, TS_2)

    # 模式分析
    patterns1, _, slopes1 = pattern_analysis(TS_1_eq)
    patterns2, _, slopes2 = pattern_analysis(TS_2_eq)

    # 绘制图表
    #plot_time_series(TS_1, TS_2, TS_1_eq, TS_2_eq, "时间序列分析", patterns1, patterns2, slopes1, slopes2)

    # 计算相似度
    similarity = calculate_similarity(TS_1_eq, TS_2_eq)
    # print(f"两个序列的形状相似度: {similarity}")
    return similarity

def plot_time_series(ts1, ts2, ts1_eq, ts2_eq, title, patterns1=None, patterns2=None, slopes1=None, slopes2=None):
    # 1. 原始的两个序列
    plt.figure(figsize=(12, 6))
    dates1 = [parse_datetime(t) for _, t in ts1]
    dates2 = [parse_datetime(t) for _, t in ts2]
    values1 = [v for v, _ in ts1]
    values2 = [v for v, _ in ts2]

    # 为TS_1添加数据点、值和模式标注
    for i, (value, date) in enumerate(zip(values1, dates1)):
        plt.plot(date, value, 'bo')
        plt.annotate(f'{value:.1f}', (date, value), textcoords="offset points", xytext=(0, 10), ha='center')
    # 为TS_1添加数据点、值和模式标注
    for i, (value, date) in enumerate(zip(values2, dates2)):
        plt.plot(date, value, 'ro')
        plt.annotate(f'{value:.1f}', (date, value), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.plot(dates1, values1, 'b-o', label='TS_1')
    plt.plot(dates2, values2, 'r-o', label='TS_2')
    plt.title(f"{title} - 原始序列")
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()

    # 2. 等序列化的两个序列
    plt.figure(figsize=(12, 6))
    dates1_eq = [parse_datetime(t) for _, t in ts1_eq]
    dates2_eq = [parse_datetime(t) for _, t in ts2_eq]
    values1_eq = [v for v, _ in ts1_eq]
    values2_eq = [v for v, _ in ts2_eq]

    # 为TS_1添加数据点、值和模式标注
    for i, (value, date) in enumerate(zip(values1, dates1)):
        plt.plot(date, value, 'bo')
        plt.annotate(f'{value:.1f}', (date, value), textcoords="offset points", xytext=(0, 10), ha='center')
    # 为TS_1添加数据点、值和模式标注
    for i, (value, date) in enumerate(zip(values2, dates2)):
        plt.plot(date, value, 'ro')
        plt.annotate(f'{value:.1f}', (date, value), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.plot(dates1_eq, values1_eq, 'b-o', label='TS_1 (等序列化)')
    plt.plot(dates2_eq, values2_eq, 'r-o', label='TS_2 (等序列化)')
    plt.title(f"{title} - 等序列化序列")
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()

    # 3. 等序列化的序列中的每条边上标记斜率
    plt.figure(figsize=(12, 6))
    plt.plot(dates1_eq, values1_eq, 'b-o', label='TS_1 (等序列化)')
    plt.plot(dates2_eq, values2_eq, 'r-o', label='TS_2 (等序列化)')

    for i in range(len(slopes1)):
        mid_point = dates1_eq[i] + (dates1_eq[i + 1] - dates1_eq[i]) / 2
        plt.annotate(f'{slopes1[i]:.1f}', (mid_point, (values1_eq[i] + values1_eq[i + 1]) / 2), textcoords="offset points", xytext=(0, 10), ha='center', color='black', fontsize=12,  fontweight='bold')

    for i in range(len(slopes2)):
        mid_point = dates2_eq[i] + (dates2_eq[i + 1] - dates2_eq[i]) / 2
        plt.annotate(f'{slopes2[i]:.1f}', (mid_point, (values2_eq[i] + values2_eq[i + 1]) / 2), textcoords="offset points", xytext=(0, -10), ha='center', color='black', fontsize=12,  fontweight='bold')

    plt.title(f"{title} - 等序列化序列（带斜率标记）")
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()

    # 4. 基于斜率算出的模式值
    plt.figure(figsize=(12, 6))
    plt.plot(dates1_eq[1:-1], patterns1, 'b-o', label='TS_1 模式')
    plt.plot(dates2_eq[1:-1], patterns2, 'r-o', label='TS_2 模式')

    for i, (pattern, date) in enumerate(zip(patterns1, dates1_eq[1:-1])):
        plt.annotate(f'{pattern}', (date, pattern), textcoords="offset points", xytext=(0, 15), ha='center', color='blue', fontsize=12,  fontweight='bold')

    for i, (pattern, date) in enumerate(zip(patterns2, dates2_eq[1:-1])):
        plt.annotate(f'{pattern}', (date, pattern), textcoords="offset points", xytext=(0, -15), ha='center', color='red', fontsize=12,  fontweight='bold')

    plt.title(f"{title} - 模式值")
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()


#### Sahpe Similarity End ####

#### Peroid Similarity Start ####

def preprocess_timeseries(ts):
    """
    预处理时间序列数据

    参数:
    ts: 输入的时间序列，格式为[(value, timestamp), ...]

    返回:
    values: 数值数组
    timestamps: 对应的datetime对象数组
    """
    values, timestamps = zip(*ts)
    values = np.array(values)
    timestamps = np.array([datetime.strptime(t, "%Y-%m-%d %H:%M") for t in timestamps])
    return values, timestamps


def compute_period_similarity(ts1, ts2, fmin=0.1, fmax=10, n_freqs=1000):
    """
    计算两个不规则采样时间序列的周期相似性

    参数:
    ts1, ts2: 两个输入的时间序列
    fmin, fmax: 频率范围的最小值和最大值
    n_freqs: 要计算的频率点数

    返回:
    similarity: 周期相似性得分
    frequencies: 频率数组
    power1_norm, power2_norm: 归一化的功率谱密度
    timestamps1, timestamps2: 原始时间戳
    values1, values2: 原始数值
    """
    # 预处理时间序列
    values1, timestamps1 = preprocess_timeseries(ts1)
    values2, timestamps2 = preprocess_timeseries(ts2)

    # 将时间戳转换为相对秒数
    timestamps1_seconds = np.array([(t - timestamps1[0]).total_seconds() for t in timestamps1])
    timestamps2_seconds = np.array([(t - timestamps2[0]).total_seconds() for t in timestamps2])

    # 计算时间序列的特性
    duration1 = timestamps1_seconds[-1] - timestamps1_seconds[0]
    duration2 = timestamps2_seconds[-1] - timestamps2_seconds[0]

    min_interval1 = np.min(np.diff(timestamps1_seconds))
    min_interval2 = np.min(np.diff(timestamps2_seconds))



    # 根据时间序列特性调整频率范围
    fmin = max(fmin, 1 / max(duration1, duration2))
    fmax = min(fmax, 1 / (2 * min(min_interval1, min_interval2)))


    # 创建频率数组
    frequencies = np.linspace(fmin, fmax, n_freqs)

    # 计算Lomb-Scargle周期图
    power1 = signal.lombscargle(timestamps1_seconds, values1, frequencies)
    power2 = signal.lombscargle(timestamps2_seconds, values2, frequencies)

    # 归一化功率谱
    power1_norm = power1 / np.sum(power1)
    power2_norm = power2 / np.sum(power2)

    # 计算余弦相似度作为周期相似性得分
    similarity = np.dot(power1_norm, power2_norm) / (np.linalg.norm(power1_norm) * np.linalg.norm(power2_norm))

    # 将相似性转换为不相似性
    dissimilarity = 1 - similarity

    # return dissimilarity, frequencies, power1_norm, power2_norm, timestamps1, timestamps2, values1, values2      #用于测试
    return dissimilarity


def plot_analysis(ts1, ts2):
    """
    绘制时间序列分析图表

    参数:
    ts1, ts2: 两个输入的时间序列
    """
    dissimilarity, frequencies, power1, power2, timestamps1, timestamps2, values1, values2 = compute_period_similarity(ts1, ts2)

    # 创建3x1的子图布局
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))

    # 绘制原始时间序列
    axs[0].plot(timestamps1, values1, label='时间序列1')
    axs[0].plot(timestamps2, values2, label='时间序列2')
    axs[0].set_xlabel('时间')
    axs[0].set_ylabel('数值')
    axs[0].set_title('原始时间序列')
    axs[0].legend()
    axs[0].grid(True)

    # 绘制功率谱密度图
    axs[1].plot(frequencies, power1, label='时间序列1')
    axs[1].plot(frequencies, power2, label='时间序列2')
    axs[1].set_xlabel('频率 (Hz)')
    axs[1].set_ylabel('归一化功率')
    axs[1].set_title('功率谱密度')
    axs[1].legend()
    axs[1].grid(True)

    # 绘制功率谱密度差异图
    axs[2].plot(frequencies, power1 - power2)
    axs[2].set_xlabel('频率 (Hz)')
    axs[2].set_ylabel('功率差异')
    axs[2].set_title('功率谱密度差异')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    print(f"周期相似性得分: {1-dissimilarity:.4f}")

#### Peroid Similarity End ####

#### Regionalization Start ####

class TimeSeriesCluster:
    """
    时间序列簇类，用于表示和管理一个时间序列簇
    """
    def __init__(self, id: str, data: List[Tuple[float, str]]):
        """
        初始化时间序列簇
        :param id: 簇的唯一标识符
        :param data: 时间序列数据，格式为 [第1个时序：[(值, 时间字符串), ...]，第2个时序：[(值, 时间字符串), ...]]
        """
        self.id = id
        self.data = data
        self.members = id.split('_')  # 簇包含的原始序列ID列表

    def merge(self, other: 'TimeSeriesCluster') -> 'TimeSeriesCluster':
        """
        将当前簇与另一个簇合并
        :param other: 要合并的另一个簇
        :return: 合并后的新簇
        """
        new_id = '_'.join(sorted(self.id.split('_') + other.id.split('_'), key= lambda item : int(item)))
        new_data = self.data + other.data
        return TimeSeriesCluster(new_id, new_data)
    

def hierarchical_clustering(similarity_measuring: str, time_series_dict: Dict[str, List[Tuple[float, str]]],
                            neighbor_dict: Dict[str, List[str]], neighbor_dict_raw: Dict[str, List[str]], k: int = 10, min_regions: int = 5) -> Dict[str, List[Tuple[float, str]]]:
    """
    执行时间序列的层次聚类，使用缓存来避免重复计算相似度。

    :param similarity_measuring: 相似性度量方法的名称（字符串形式）
    :param time_series_dict: 时间序列字典，键是区域ID，值是由(值, 时间字符串)元组组成的列表
    :param neighbor_dict: 邻居信息字典，键是区域ID，值是邻居区域ID列表
    :param k: 聚类停止的阈值，默认为10
    :return: 聚类后的时间序列字典
    """
    # 初始化簇，每个时间序列作为一个独立的簇
    clusters = {id: TimeSeriesCluster(id, [data]) for id, data in time_series_dict.items()}
    similarity_cache = {}  # 用于缓存已计算的相似度

    def get_similarity(cluster1: TimeSeriesCluster, cluster2: TimeSeriesCluster) -> float:
        """
        计算两个簇之间的相似度
        :param cluster1: 第一个簇
        :param cluster2: 第二个簇
        :return: 相似度值
        """
        key = tuple(sorted([cluster1.id, cluster2.id]))
        if key not in similarity_cache:
            if len(cluster1.members) == 1 and len(cluster2.members) == 1:
                # 如果两个簇都只包含一个原始序列，直接计算相似度
                similarity_cache[key] = round(eval(similarity_measuring)(cluster1.data[0], cluster2.data[0]),6)
            else:
                # 如果至少有一个簇包含多个序列，随机选择序列计算平均相似度
                similarities = []
                series1 = cluster1.data if len(cluster1.members) == 1 else random.sample(cluster1.data, min(10, len(cluster1.data)))
                series2 = cluster2.data if len(cluster2.members) == 1 else random.sample(cluster2.data, min(10, len(cluster2.data)))
                for s1 in series1:
                    for s2 in series2:
                        similarities.append(eval(similarity_measuring)(s1, s2))
                similarity_cache[key] = sum(similarities) / len(similarities)
        return similarity_cache[key]

    def should_force_merge(cluster1, cluster2, neighbor_dict_raw):
        """
        判断是否应该强制合并两个簇。

        :param cluster1: 第一个簇
        :param cluster2: 第二个簇
        :param neighbor_dict_raw: 存储每个区域原始邻居信息的字典

        :return: 如果应该强制合并，则返回True；否则返回False
        """
        # 检查cluster1的所有邻居是否都包含在cluster2中
        # 如果是，说明cluster1被cluster2完全包围
        cluster1_neighbors = []
        for item in cluster1.id.split('_'):
            cluster1_neighbors.extend(neighbor_dict_raw[item])
        if set(cluster1_neighbors) - set(cluster1.id.split('_')) <= set(cluster2.id.split('_')):
            return True
        # 如果不满足上述任何条件，则不进行强制合并
        return False

    def count_regions(cluster):
        return len(cluster.id.split('_'))

    # 初始化相似度堆，存储所有相邻簇对的相似度
    similarity_heap = []
    for id1 in clusters:
        for id2 in neighbor_dict.get(id1, []):
            if id2 in clusters and int(id1) < int(id2):
                similarity = get_similarity(clusters[id1], clusters[id2])
                heapq.heappush(similarity_heap, (similarity, id1, id2))  # 使用负的相似度，因为heapq是最小堆

        # 主循环：不断合并最相似的簇，直到达到停止条件
    while len(clusters) > k:
        # 首先检查是否有全包围的情况
        force_merge = None
        for id1 in clusters:
            for id2 in neighbor_dict[id1]:
                if id2 in clusters and should_force_merge(clusters[id1], clusters[id2], neighbor_dict_raw):
                    force_merge = (id1, id2, None)
                    break
            if force_merge:
                break

        # 如果没有全包围情况，检查是否有小于min_regions的簇需要合并
        if not force_merge and len(clusters) < 2 * k:
            small_clusters = [id for id, cluster in clusters.items() if count_regions(cluster) < min_regions]
            if small_clusters:
                id1 = small_clusters[0]
                # 找到与小簇最相似的邻居
                max_similarity = float('inf')
                best_neighbor = None
                for id2 in neighbor_dict[id1]:
                    if id2 in clusters:
                        similarity = get_similarity(clusters[id1], clusters[id2])
                        if similarity < max_similarity:
                            max_similarity = similarity
                            best_neighbor = id2
                if best_neighbor:
                    force_merge = (id1, best_neighbor, round(max_similarity, 5))

        if force_merge:
            id1, id2, max_similarity = force_merge
            cluster1, cluster2 = clusters[id1], clusters[id2]
            reason = "全包围" if should_force_merge(cluster1, cluster2,
                                                    neighbor_dict_raw) else f"簇大小 ({count_regions(cluster1)} < {min_regions})"
            print(
                f"强制合并区域 {sorted(cluster1.id.split('_'))} 到 {sorted(cluster2.id.split('_'))},相似性：{max_similarity}（{reason}）")
        else:
            # 如果没有全包围的情况，则按相似度合并
            if not similarity_heap:
                break

            max_similarity, id1, id2 = heapq.heappop(similarity_heap)
            # max_similarity = -max_similarity  # 转回正的相似度值

            if id1 not in clusters or id2 not in clusters:
                continue

            cluster1, cluster2 = clusters[id1], clusters[id2]
            print(
                f"合并区域 {sorted(cluster1.id.split('_'))} 和 {sorted(cluster2.id.split('_'))}, 相似度: {max_similarity:.6f}，当前区域数量: {len(clusters)}")

        # 合并两个最相似的相邻簇
        new_cluster = clusters[id1].merge(clusters[id2])
        clusters[new_cluster.id] = new_cluster
        del clusters[id1]
        del clusters[id2]

        # 更新邻居信息
        new_neighbors = list(set(neighbor_dict.get(id1, []) + neighbor_dict.get(id2, [])) - {id1, id2})
        neighbor_dict[new_cluster.id] = new_neighbors
        for neighbor in new_neighbors:
            if neighbor in neighbor_dict:
                neighbor_dict[neighbor] = [new_cluster.id if r in (id1, id2) else r for r in neighbor_dict[neighbor]]

        # 计算新簇与其邻居的相似度并更新堆
        for neighbor in new_neighbors:
            if neighbor in clusters:
                similarity = get_similarity(new_cluster, clusters[neighbor])
                heapq.heappush(similarity_heap,
                               (similarity, min(new_cluster.id, neighbor), max(new_cluster.id, neighbor)))

        # 返回最终的聚类结果
    return {cluster.id: cluster.data for cluster in clusters.values()}

# 可视化给定簇中的时序数据
def visualize_clusters(time_series_dict, clusters):
    """
    可视化给定簇中的时序数据

    参数:
    time_series_dict (dict): 以区域 ID 为键,时序数据为值的字典
    clusters (list): 包含簇中区域 ID 列表的列表
    """
    num_clusters = len(clusters)

    # 创建包含 num_clusters 个子图的figure
    fig, axs = plt.subplots(num_clusters, 1, figsize=(12, 4 * num_clusters))

    # 遍历每个簇
    for i, cluster in enumerate(clusters):
        # 绘制每个区域的时序数据
        for region_id in cluster.split('_'):
            if str(region_id) in time_series_dict:
                values, timestamps = zip(*time_series_dict[str(region_id)])
                timestamps = [datetime.strptime(t, "%Y-%m-%d %H:%M") for t in timestamps]
                axs[i].plot(timestamps, values, label=str(region_id))

        axs[i].set_title(f"簇 {i + 1}")
        axs[i].set_xlabel("时间")
        axs[i].set_ylabel("值")
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()

#### Regionalization End ####

def copy_feature_class_and_add_similarity(original_fc, new_fc, cluster_dic, region_id_field, field_name="cluster_id"):
    # 复制原始要素类
    arcpy.CopyFeatures_management(original_fc, new_fc)
    
    # 在新要素类中添加相似度字段
    arcpy.AddField_management(new_fc, field_name, "DOUBLE")
    arcpy.AddMessage("###########")

    # 更新字段值
    with arcpy.da.UpdateCursor(new_fc, [region_id_field, field_name]) as cursor:
        for row in cursor:
            region_id = row[0]
            arcpy.AddMessage(region_id)
            if region_id in cluster_dic:
                row[1] = cluster_dic[region_id]
                cursor.updateRow(row)
                arcpy.AddMessage("###########")
                arcpy.AddMessage(str(row[0]) + ": " + str(row[1]))
    
    return new_fc

def featureclass_symbolize(output_fc, symbol_field):
    try:

        # 获取当前地图
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        map = aprx.activeMap

        # 创建图层并添加到地图
        layer = map.addDataFromPath(output_fc)
        arcpy.AddMessage("已将新图层添加到当前地图")

        # 检查 'cluster_id' 字段是否存在
        field_names = [field.name for field in arcpy.ListFields(output_fc)]
        if symbol_field not in field_names:
            arcpy.AddWarning( str(symbol_field) + " 字段不存在，无法进行符号化")
            return

        # 应用唯一值符号化
        sym = layer.symbology
        if hasattr(sym, 'renderer'):
            sym.updateRenderer('UniqueValueRenderer')
            sym.renderer.fields = [symbol_field]
            layer.symbology = sym
            arcpy.AddMessage("已应用基于 'cluster_id' 的唯一值符号化")
        else:
            arcpy.AddWarning("无法应用符号化，可能是因为图层类型不支持")

    except arcpy.ExecuteError:
        arcpy.AddError(arcpy.GetMessages(2))
    except Exception as e:
        arcpy.AddError(f"发生错误: {str(e)}")

def script_tool(input_areal_featurelayer, region_id_field, sts_param0, neighborMetrix_param1, similarity_method, region_num,  regionalizaiton_result_featureclass, minimum_number = 5):
    """Script code goes below"""

    time_series_filepath = sts_param0        #值相似性测试纵向，周期相似性纵向， 形状相似性测试纵向 periodic_sequences
    time_series_dict = create_lengthwise_time_series_list(time_series_filepath)

    # 邻居数据
    nei_file_path = neighborMetrix_param1
    neighbor_dict = create_crosswise_neighbor_dict(nei_file_path)
    neighbor_dict_raw = copy.deepcopy(neighbor_dict)

    # 相似性度量方法选择
    method = ['coumput_dtw_similarity', 'compute_Edis_similarity', 'calculate_shape_similarity','compute_period_similarity']

    result = {}
    if similarity_method == "European_Distance_based_Value_Similairty":
        result = hierarchical_clustering(method[0], time_series_dict, neighbor_dict, neighbor_dict_raw, k=int(region_num), min_regions=int(minimum_number))# k结果中簇的数量，min_regions：每个簇中预期最少区域的数量（算法会尽量满足）
    if similarity_method == "DTW_based_Value_Similarity":
        result = hierarchical_clustering(method[1], time_series_dict, neighbor_dict, neighbor_dict_raw, k=int(region_num), min_regions=int(minimum_number))# k结果中簇的数量，min_regions：每个簇中预期最少区域的数量（算法会尽量满足）
    if similarity_method == "Cosine_based_Shape_Similarity":
        result = hierarchical_clustering(method[2], time_series_dict, neighbor_dict, neighbor_dict_raw, k=int(region_num), min_regions=int(minimum_number))# k结果中簇的数量，min_regions：每个簇中预期最少区域的数量（算法会尽量满足）
    if similarity_method == "Fourier_based_Period_Similarity":
        result = hierarchical_clustering(method[3], time_series_dict, neighbor_dict, neighbor_dict_raw, k=int(region_num), min_regions=int(minimum_number))# k结果中簇的数量，min_regions：每个簇中预期最少区域的数量（算法会尽量满足）

    # 打印结果
    for one_cluster in result.keys():
        arcpy.AddMessage(f"{sorted([int(id) for id in one_cluster.split('_')])}")

    new_dict = {}
    for index, one_cluster in enumerate(result.keys(), start=1):
        sorted_ids = sorted([int(id) for id in one_cluster.split('_')])
        for id in sorted_ids:
            new_dict[id] = index
        # arcpy.AddMessage(f"List {index}: {sorted_ids}")

    arcpy.AddMessage(new_dict)

    copy_feature_class_and_add_similarity(input_areal_featurelayer, regionalizaiton_result_featureclass, new_dict, region_id_field)

    featureclass_symbolize(regionalizaiton_result_featureclass, "cluster_id")

    return


if __name__ == "__main__":

    input_areal_featurelayer = arcpy.GetParameterAsText(0)
    region_id_field = arcpy.GetParameterAsText(1)
    sts_param0 = arcpy.GetParameterAsText(2)
    neighborMetrix_param1 = arcpy.GetParameterAsText(3)
    similarity_method = arcpy.GetParameterAsText(4)
    region_num = arcpy.GetParameterAsText(5)
    minimum_number = arcpy.GetParameterAsText(6)
    regionalizaiton_result_featureclass = arcpy.GetParameterAsText(7)

    arcpy.AddMessage("ok")

    script_tool(input_areal_featurelayer, region_id_field, sts_param0, neighborMetrix_param1, similarity_method, region_num,  regionalizaiton_result_featureclass, minimum_number)
   