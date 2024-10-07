import datetime
from typing import List, Tuple
import csv
import arcpy
import os
import tempfile
import pandas as pd


# 从给定的文件路径读取时序信息，创建并返回时序列表。
def create_lengthwise_time_series_list_by_specific_column(file_path, new_file_path, id_col=0, value_col=1, date_col=-1, scale_rate=1):
    """
    从给定的文件路径读取时序信息，创建并返回时序列表。

    参数:
    file_path (str): 包含时序信息的文件路径
    id_col (int): 指定id所在的列
    value_col (int): 指定值所在的列
    date_col (int): 指定日期所在的列
    scale_rate (int): 指定缩放比例

    返回:
    list: 包含所有时序序列的列表，每个序列是一个列表的元组 (值, 时间字符串)
    """

    arcpy.AddMessage(file_path)

    # 初始化一个字典来存储所有的时序序列
    series_dict = {}

    # 设置起始时间（用于两列数据的情况）
    start_time = datetime.datetime(2024, 1, 1, 0, 0)

    try:
        # 尝试打开并读取文件
        with open(file_path, 'r') as file:
            # 逐行读取文件内容
            next(file)
            for line in file:
                # 去除行首尾的空白字符，并按逗号分割

                items = line.strip().split(',')

                # 如果分割后的列表少于2个或多于3个元素，则跳过此行
                if len(items) < 2 or len(items) > 3:
                    print(f"警告: 时序文件跳过格式不正确的行: {line.strip()}")
                    continue

                # 提取序列编号和值
                sequence_number = items[id_col]
                value = float(items[value_col])  # 将值转换为小数

                # 处理时间信息
                if date_col != -1:
                    # 如果是三列数据，直接使用提供的时间
                    time_str = items[date_col]
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
        arcpy.AddMessage(f"错误：未找到文件 '{file_path}'。")
        return []
    except IOError:
        # 如果文件无法读取（比如权限问题），打印错误信息
        arcpy.AddMessage(f"错误：无法读取文件 '{file_path}'。")
        return []
    except IOError as e:     #ValueError
        # 如果在转换过程中出现错误（例如，非整数值），打印错误信息
        arcpy.AddMessage(f"错误：处理数据时出现问题 - {str(e)}")
        return []

    # 根据给定的压缩比压缩时间序列。
    scaled_series_dict = {}
    for index, time_series in series_dict.items():
        scaled_time_series = compress_time_series(time_series, scale_rate)
        scaled_series_dict[index] = scaled_time_series

    arcpy.AddMessage("######")
    #arcpy.AddMessage(scaled_series_dict)

    write_dict_to_csv(scaled_series_dict, new_file_path)




# 根据给定的压缩比压缩时间序列。
def compress_time_series(ts: List[Tuple[int, str]], scale_rate_str: float) -> List[Tuple[int, str]]:
    """
    根据给定的压缩比压缩时间序列。

    :param ts: 原始时间序列，格式为 (值, 时间戳) 的元组列表
    :param scale_rate: 压缩比 (0 < scale_rate <= 1)
    :return: 压缩后的时间序列
    """
    scale_rate = float(scale_rate_str)

    # 检查压缩比是否有效
    if not (0 < scale_rate <= 1):
        raise ValueError("压缩比 scale_rate 必须在 0 和 1 之间")

    # 如果时间序列长度小于等于2，无需压缩，直接返回
    if len(ts) <= 2:
        return ts

    # 计算需要保留的元素数量
    n = len(ts)
    keep_count = max(2, int(n * scale_rate))

    # 始终保留第一个元素
    result = [ts[0]]

    if keep_count > 2:
        # 计算选择元素的步长
        step = (n - 2) / (keep_count - 2)

        # 对中间的元素进行插值
        for i in range(1, keep_count - 1):
            index = int(i * step)
            prev_index = int((i - 1) * step)
            next_index = min(int((i + 1) * step), n-1)
            # 插值计算新的值
            value = interpolate_value(ts[prev_index][0], ts[index][0], ts[next_index][0])

            # 插值计算新的时间戳
            timestamp = interpolate_timestamp(ts[prev_index][1], ts[index][1], ts[next_index][1])

            result.append((value, timestamp))

    # 添加最后一个元素
    result.append(ts[-1])
    return result

# 基于周围的时间戳插值计算新的时间戳。
def interpolate_value(prev: float, current: float, next: float) -> float:
    """
    基于周围的值插值计算新的值。

    :param prev: 前一个值
    :param current: 当前值
    :param next: 后一个值
    :return: 插值后的新值
    """
    return round(float((prev + current + next) / 3), 3)

# 基于周围的时间戳插值计算新的时间戳。
def interpolate_timestamp(prev: str, current: str, next: str) -> str:
    """
    基于周围的时间戳插值计算新的时间戳。

    :param prev: 前一个时间戳
    :param current: 当前时间戳
    :param next: 后一个时间戳
    :return: 插值后的新时间戳
    """
    # 将字符串时间戳转换为 datetime 对象
    prev_dt = datetime.datetime.strptime(prev, "%Y-%m-%d %H:%M")
    current_dt = datetime.datetime.strptime(current, "%Y-%m-%d %H:%M")
    next_dt = datetime.datetime.strptime(next, "%Y-%m-%d %H:%M")

    # 计算插值后的时间戳
    interpolated_dt = prev_dt + (next_dt - prev_dt) / 2
    return interpolated_dt.strftime("%Y-%m-%d %H:%M")

# 将标准化后的时间序列字典数据写入CSV文件。
def write_dict_to_csv(scaled_time_series_dict, output_file):
    """
    将嵌套字典数据写入CSV文件。

    :param scaled_time_series_dict: 包含时间序列数据的字典
    :param output_file: 输出CSV文件的名称
    """
    # 打开CSV文件进行写入
    with open(output_file, 'w', newline='') as csvfile:
        # 创建CSV writer对象
        csv_writer = csv.writer(csvfile)

        # 写入CSV头部
        # csv_writer.writerow(['id', '值', '日期'])

        # 遍历字典中的每个时间序列
        for id, time_series in scaled_time_series_dict.items():
            # 遍历每个时间序列中的数据点
            for value, timestamp in time_series:
                # 写入一行数据
                csv_writer.writerow([id, value, timestamp])


def read_csv_first_row(csv_file_path):
    """
    读取CSV文件的第一行，并返回一个包含这些值的列表。
    
    参数:
    csv_file_path (str): CSV文件的路径
    
    返回:
    list: 包含CSV文件第一行值的列表，如果文件为空则返回None
    """
    try:
        with open(csv_file_path, 'r', newline='') as file:
            csv_reader = csv.reader(file)
            first_row = next(csv_reader, None)
        return first_row
    except FileNotFoundError:
        print(f"错误：找不到文件 '{csv_file_path}'")
        return None
    except PermissionError:
        print(f"错误：没有权限读取文件 '{csv_file_path}'")
        return None
    except Exception as e:
        print(f"读取文件时发生错误：{str(e)}")
        return None

def replace_filename(original_path, new_filename):
    """
    替换给定文件路径中的文件名，保留原始目录和扩展名。

    参数:
    original_path (str): 原始文件的完整路径
    new_filename (str): 新的文件名（不包括扩展名）

    返回:
    str: 新的完整文件路径
    """
    # 分离文件路径、文件名和扩展名
    directory, old_filename = os.path.split(original_path)
    _, extension = os.path.splitext(old_filename)

    # 确保新文件名不包含扩展名
    new_filename = os.path.splitext(new_filename)[0]

    # 创建新的文件名（新名称加上原始扩展名）
    new_full_filename = new_filename + extension

    # 组合新的完整文件路径
    new_path = os.path.join(directory, new_full_filename)

    return new_path


def get_csv_field_indices(csv_path, field_a, field_b, field_c):
    """
    从CSV文件中获取指定字段的列索引。

    参数:
    csv_path (str): CSV文件的路径
    field_a (str): 第一个字段名
    field_b (str): 第二个字段名
    field_c (str): 第三个字段名

    返回:
    tuple: 包含三个字段在CSV中的列索引（基于0的索引）
    """

    try:
        df = pd.read_csv(csv_path, encoding='utf-8')  # 或者尝试 'gbk', 'utf-8-sig' 等
        headers = df.columns.tolist()

        #arcpy.AddMessage(headers)
        # 获取指定字段的索引
        try:
            index_a = headers.index(field_a)
            index_b = headers.index(field_b)
            index_c = headers.index(field_c)
            #arcpy.AddMessage(index_a)
        except ValueError as e:
            #arcpy.AddError(f"未找到指定的字段: {str(e)}")
            return None

        return (index_a, index_b, index_c)

    except Exception as e:
        arcpy.AddError(f"读取CSV文件时发生错误: {str(e)}")
        return None

def script_tool(input_file, id_field, value_field, datetime_field, Scale_rate, output_file):
    """Script code goes below"""

    desc = arcpy.Describe(input_file)
    full_path = desc.catalogPath
    arcpy.AddMessage(full_path)

    fieldIndexList = get_csv_field_indices(full_path, id_field, value_field, datetime_field)

    arcpy.AddMessage(fieldIndexList)


    create_lengthwise_time_series_list_by_specific_column(full_path, output_file, fieldIndexList[0], fieldIndexList[1], -1, Scale_rate)

    return


if __name__ == "__main__":

    input_file = arcpy.GetParameterAsText(0)
    id_field = arcpy.GetParameterAsText(1)
    value_field = arcpy.GetParameterAsText(2)
    datetime_field = arcpy.GetParameterAsText(3)
    Scale_rate = arcpy.GetParameterAsText(4)
    output_file = arcpy.GetParameterAsText(5)

    script_tool(input_file, id_field, value_field, datetime_field, Scale_rate, output_file)