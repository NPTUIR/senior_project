from . import setpath
import numpy as np
setpath.path_process()
print('init path')

def average_filter(data, window_size):
    """
    平均濾波器
    :param data: 要進行濾波的數據
    :param window_size: 滑動窗口的大小
    :return: 濾波後的數據
    """
    filtered_data = []
    for i in range(len(data)):
        if i < window_size - 1:
            filtered_data.append(data[i])
        else:
            filtered_data.append(sum(data[i - window_size + 1:i + 1]) / window_size)
    return filtered_data

def median_filter(data, window_size):
    half_window_size = window_size // 2
    filtered_data = np.zeros_like(data)
    for i in range(half_window_size, len(data) - half_window_size):
        filtered_data[i] = np.median(data[i - half_window_size:i + half_window_size + 1])
    return filtered_data

def smooth(data):
    smooth_data=[]
    for i in data:
        smooth_data.append(i)
        d = sum(smooth_data)/len(smooth_data)
        smooth_data[-1]=d
    return smooth_data