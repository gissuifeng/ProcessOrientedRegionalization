import arcpy
import csv
import math
import os
import numpy as np
import bisect
import datetime
from scipy import signal

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
    start_time = datetime.datetime(2024, 1, 1, 0, 0)

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
                    print(f"警告: 时序文件跳过格式不正确的行: {line.strip()}")
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
                        last_time = datetime.datetime.strptime(series_dict[sequence_number][-1][1], "%Y-%m-%d %H:%M")
                        current_time = last_time + datetime.timedelta(minutes=5)

                    time_str = current_time.strftime("%Y-%m-%d %H:%M")

                # 将 (值, 时间) 元组添加到对应的序列
                if sequence_number in series_dict:
                    series_dict[sequence_number].append((value, time_str))
                else:
                    series_dict[sequence_number] = [(value, time_str)]

    except FileNotFoundError:
        # 如果文件不存在，打印错误信息
        print(f"错误：未找到文件 '{file_path}'。")
        return []
    except IOError:
        # 如果文件无法读取（比如权限问题），打印错误信息
        print(f"错误：无法读取文件 '{file_path}'。")
        return []
    except ValueError as e:     #ValueError
        # 如果在转换过程中出现错误（例如，非整数值），打印错误信息
        print(f"错误：处理数据时出现问题 - {str(e)}")
        return []

    # 将字典返回
    return series_dict


#### Value Similarity Start ####

def calculate_similarity(series1, series2):
    """
    计算两个时间序列的相似度
    :param series1: 第一个时间序列
    :param series2: 第二个时间序列
    :return: 相似度值
    """
    # 计算欧式距离
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(series1, series2)))
    
    # 计算相似度（使用1 / (1 + 距离)，使得距离越小，相似度越大）
    similarity = 1 / (1 + distance)
    
    return similarity

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

    # 计算相似度（使用1 / (1 + 距离)，使得距离越小，相似度越大）
    total_similarity = total_similarity

    return 1 / (1 + total_similarity)


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
                print(e)
            # 选择最小成本路径
            cost_matrix[i, j] = cost + min(cost_matrix[i - 1, j],  # 插入
                                           cost_matrix[i, j - 1],  # 删除
                                           cost_matrix[i - 1, j - 1])  # 匹配

    # 返回右下角的值，即总的DTW距离
    return 1 / (1 + cost_matrix[n, m])

#### Value Similarity End ####

#### Shape Similarity Start ####

def parse_datetime(date_string):
    return datetime.datetime.strptime(date_string, "%Y-%m-%d %H:%M")

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
    # plot_time_series(TS_1, TS_2, TS_1_eq, TS_2_eq, "时间序列分析", patterns1, patterns2, slopes1, slopes2)

    # 计算相似度
    similarity = calculate_similarity(TS_1_eq, TS_2_eq)
    # print(f"两个序列的形状相似度: {similarity}")

    return 1 / (1 + similarity)

#### Shape Similarity End ####


### Peroid Similarity Start ###

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
    timestamps = np.array([datetime.datetime.strptime(t, "%Y-%m-%d %H:%M") for t in timestamps])
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
    return 1 / (1 + dissimilarity)

### Peroid Similarity Start ###

def process_csv(file_path, target_region_id, similarity_method):
    # 读取CSV文件
    data = create_lengthwise_time_series_list(file_path)
    

    # 获取目标区域的时间序列
    target_series = data[str(target_region_id)]

    # 计算相似度
    similarities = {}
    for region_id, series in data.items():
        if region_id == target_region_id:
            continue
        
        if similarity_method == "European_Distance_based_Value_Similairty":
            similarity = compute_Edis_similarity(target_series, series)

        if similarity_method == "DTW_based_Value_Similarity":
            similarity = coumput_dtw_similarity(target_series, series)

        if similarity_method == "Cosine_based_Shape_Similarity":
            similarity = calculate_shape_similarity(target_series, series)

        if similarity_method == "Fourier_based_Period_Similarity":
            similarity = compute_period_similarity(target_series, series)

        #arcpy.AddMessage("#####")
        #arcpy.AddMessage(str(region_id) + "_" + str(target_region_id) + ":" + str(similarity))
        similarities[region_id] = similarity

    similarities[target_region_id] = 1.0

    arcpy.AddMessage(similarities)

    return similarities



def copy_feature_class_and_add_similarity(original_fc, new_fc, similarities, field_name="Similarity"):
    # 复制原始要素类
    arcpy.CopyFeatures_management(original_fc, new_fc)
    
    # 在新要素类中添加相似度字段
    arcpy.AddField_management(new_fc, field_name, "DOUBLE")

    # 更新字段值
    with arcpy.da.UpdateCursor(new_fc, ["region_id", field_name]) as cursor:
        for row in cursor:
            region_id = str(row[0])
            arcpy.AddMessage(str(region_id))
            if region_id in similarities:
                row[1] = similarities[region_id]
                cursor.updateRow(row)
                arcpy.AddMessage("###########")
                arcpy.AddMessage(str(row[0]) + ": " + str(row[1]))
    
    return new_fc

def script_tool(area_feature_layer, area_id_field, focal_area_id,sts_data, similarity_method, output_similarity_layer):
    """Script code goes below"""

    target_region_id = int(focal_area_id)

    similarities = process_csv(sts_data, target_region_id, similarity_method)
    #arcpy.AddMessage(similarities)

    # 将相似度添加到面要素类
    copy_feature_class_and_add_similarity(area_feature_layer, output_similarity_layer, similarities)
    arcpy.AddMessage(f"新的要素类 '{os.path.basename(output_similarity_layer)}' 已创建，并添加了相似度列。")

    return


if __name__ == "__main__":

    arcpy.env.overwriteOutput = True

    area_feature_layer = arcpy.GetParameterAsText(0)
    area_id_field = arcpy.GetParameterAsText(1)
    focal_area_id = arcpy.GetParameterAsText(2)
    sts_data = arcpy.GetParameterAsText(3)
    similarity_method = arcpy.GetParameterAsText(4)
    output_similarity_layer = arcpy.GetParameterAsText(5)
    

    script_tool(area_feature_layer, area_id_field, focal_area_id,sts_data, similarity_method, output_similarity_layer)
