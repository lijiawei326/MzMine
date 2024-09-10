import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import random
from sympy import Point, Line
import os


def score(p1,p2):
    mz1,rt1 = p1
    mz2,rt2 = p2
    mz_weight = 1
    rt_weight = 1
    score = np.sqrt(((mz1-mz2)**2) * mz_weight + ((rt1-rt2)**2) * rt_weight)
    return score


def best_search(family,peaks):
    center_mz = family['mean']['mz']
    center_rt = family['mean']['rt']
    candidate = []
    for i in peaks:
        if np.abs(center_mz - i[0]) <= 1 and np.abs(center_rt -i[1]) <= 2:
            candidate.append(i)
    best_dist = np.inf
    if len(candidate) == 0:
        return []
    for c in candidate:
        s = score((center_mz,center_rt),(c[0],c[1]))
        if s < best_dist:
            best_dist = s
            best_match = c

    return [best_match]


def candidate_search(family,peaks):
    center_mz = family['mean']['mz']
    center_rt = family['mean']['rt']
    candidate = []
    for i in peaks:
        if np.abs(center_mz - i[0]) <= 1 and np.abs(center_rt -i[1]) <= 2:
            candidate.append(i)
    return candidate


def linear_regression(list_x, list_y):
    list_x = np.array(list_x)
    list_y = np.array(list_y)
    size = len(list_x)  # 数据点的总数
    OUT = size*0.6    

    # 进行迭代操作，寻找最佳的线性回归模型参数a和b
    iters = 1000  # 迭代次数
    epsilon = 0.1  # 内点的误差阈值
    threshold = (size - OUT) / size + 0.01  # 阈值，用于控制早停（early stopping）策略
    best_a, best_b = 0, 0  # 最佳线性回归模型的参数，初始值为0
    pre_total = 0  # 内点数量的初始值，初始为0

    # 进行迭代操作，寻找最佳的线性回归模型参数a和b
    for i in range(iters):
        # 从size个数据点中随机选择两个点，索引存储在sample_index中
        sample_index = random.sample(range(size), 2)
        x_1 = list_x[sample_index[0]]  # 获取第一个点的x值
        x_2 = list_x[sample_index[1]]  # 获取第二个点的x值
        y_1 = list_y[sample_index[0]]  # 获取第一个点的y值
        y_2 = list_y[sample_index[1]]  # 获取第二个点的y值

        # 根据两个点的坐标计算出线性回归模型的斜率a和截距b
        if (x_2 - x_1) == 0:
            continue
        a = (y_2 - y_1) / (x_2 - x_1)  # 计算斜率a
        b = y_1 - a * x_1  # 计算截距b
        total_in = 0  # 内点计数器，初始值为0

        # 对于每一个数据点，计算其对应的预测值，并与真实值进行比较，如果误差小于epsilon，则认为此点为内点，计数器加1
        for index in range(size):
            y_estimate = a * list_x[index] + b  # 根据线性回归模型计算出预测值
            if abs(y_estimate - list_y[index]) < epsilon:  # 判断预测值与真实值的误差是否小于epsilon
                total_in += 1  # 如果小于epsilon，则此点为内点，计数器加1

        # 如果当前的内点数量大于之前所有的内点数量，则更新最佳参数a和b，以及内点数量pre_total
        if total_in > pre_total:  # 记录最大内点数与对应的参数
            pre_total = total_in
            best_a = a
            best_b = b

        # 如果当前的内点数量大于设定的阈值所对应的人数，则跳出循环，不再进行迭代
        if total_in > size * threshold:  # 如果当前内点数量大于阈值所设定的人数，则跳出循环
            break  # 跳出循环
    print("迭代{}次,a = {}, b = {}".format(i, best_a, best_b))  # 输出当前迭代的次数，以及对应的线性回归模型参数a和b
    line = Line((0,best_b),slope=best_a)
    return line


from statsmodels.regression.linear_model import WLS
from statsmodels.tools.tools import add_constant
from scipy.interpolate import PchipInterpolator

def local_weight(len):
    center = (len+1) / 2
    w = np.array([1/np.abs(i+1 - center) if i+1 != center else 2 for i in range(len)])
    return w


def loess(point_list):
    point_list.sort(key=lambda x:x[0])
    frac = 10
    list_x,list_y = np.array([p[0] for p in point_list]),np.array([p[1] for p in point_list])
    loess_point = []
    uni_list_x = np.unique(list_x)
    for i in range(len(uni_list_x)):
        rt_x = uni_list_x[i]
        idx = (list_x > rt_x-frac) & (list_x < rt_x+frac)
        local_x = list_x[idx]  
        local_y = list_y[idx]
        w = local_weight(len(local_x))
        model_wls = WLS(local_y,local_x,w,hasconst=True).fit()
        pred_y = model_wls.predict(rt_x)
        loess_point.append((rt_x,pred_y[0]))

    pchip = PchipInterpolator([p[0] for p in loess_point],[p[1] for p in loess_point])
    return pchip

def get_all_candidate(peaks_family,peaks):
    """
    return: rt [(center, sample)]
    """
    all_candidate = []
    for family in peaks_family:
        candidate = candidate_search(family,peaks)
        center_rt = family['mean']['rt']
        all_candidate.extend([(center_rt,c[1]) for c in candidate])
    return all_candidate


def get_outlier(peaks_family,peaks):
    outlier = []
    c_rts = [f['mean']['rt'] for f in peaks_family]
    c_mzs = [f['mean']['mz'] for f in peaks_family]
    center = np.array([c_mzs,c_rts]).T
    for pk in peaks:
        p = np.array([pk])
        delta = np.abs(center - [p])[0]
        delta = delta[(delta[:,0] <= 2) & (delta[:,1] <= 4)]
        if len(delta) == 0:
            outlier.append(pk)
    if len(outlier) < 2:
        return outlier
    res = []
    for ol in outlier:
        outlier_list = outlier.copy()
        outlier_list.remove(ol)
        o = np.array([ol])
        outlier_list = np.array(outlier_list)
        delta = np.abs(outlier_list - o)
        if len(delta.shape) > 2:
            delta = delta[0]
        delta = delta[(delta[:,0] <= 2) & (delta[:,1] <= 4)]
        if len(delta) == 0:
            res.append(ol)
    return res


def update_family(family,best_match):
    assert type(best_match) == list
    family['peaks'].extend(best_match)
    family['peaks'] = list(set(family['peaks']))
    family['mean']['mz'] = np.mean([p[0] for p in list(set(family['peaks']))])
    family['mean']['rt'] = np.mean([p[1] for p in list(set(family['peaks']))])


def mzmine(peaks_family,peaks):
    all_candidate = get_all_candidate(peaks_family,peaks)
    outlier = get_outlier(peaks_family,peaks)
    print(f"候选峰数量：{len(all_candidate)}，孤立点数量:{len(outlier)}")
    line = linear_regression([i[0] for i in all_candidate], [i[1] for i in all_candidate])
    # RANSAC获取内点
    candidate_inliner = [c for c in all_candidate if line.distance((c[0],c[1])) < 0.1]
    # loess拟合
    pchip = loess(candidate_inliner)
    new_peaks = [(p[0],pchip([p[1]])[0]) for p in peaks]
    new_outlier = [(o[0],pchip([o[1]])[0]) for o in outlier]
    return new_peaks,new_outlier


def get_feature_peaks(peaks_family,name,gcms_data,gcms_data_peaks):
    new_peaks,new_outlier = mzmine(peaks_family,gcms_data_peaks)
    new_peaks_w_intensity = [(n[0],n[1],gcms_data.loc[o[1],str(int(o[0]))]) for o,n in zip(gcms_data_peaks,new_peaks)]
    feature_dict = {}
    for family in peaks_family:
        id = family['id']
        best_match = best_search(family,new_peaks_w_intensity)
        if len(best_match) == 0:
            feature_dict.update({id:np.nan})
        else:
            feature_dict.update({id:best_match[0][2]})
    return pd.DataFrame(feature_dict,index=[name])



if __name__ == "__main__":
    #########------读取数据-------########## 
    def load_peaks(file_name):
        data = np.load(os.path.join('./data_peaks',file_name))
        data = [tuple(i) for i in data]
        return data

    data_peaks_files = os.listdir('./data_peaks')
    data_peaks_files = {k.split('.')[0]+'.csv':load_peaks(k) for k in data_peaks_files}
    gcms_data_peaks = data_peaks_files


    # 初始化峰族
    peaks_family = []
    first_sample = gcms_data_peaks['酒-16_0_三轮次醇甜.csv']
    res = []
    for ol in first_sample:
        outlier_list = first_sample.copy()
        outlier_list.remove(ol)
        o = np.array([ol])
        outlier_list = np.array(outlier_list)
        delta = np.abs(outlier_list - o)
        if len(delta.shape) > 2:
            delta = delta[0]
        delta = delta[(delta[:,0] <= 2) & (delta[:,1] <= 4)]
        if len(delta) == 0:
            res.append(ol)
    first_sample = res
    for i in first_sample:
        peaks_family.append({'mean':{'mz':i[0],'rt':i[1]},'peaks':[i],'id':len(peaks_family)+1})


    for times in range(2):
        for k,v in gcms_data_peaks.items():
            new_peaks,new_outlier = mzmine(peaks_family,v)
            for family in peaks_family:
                best_match = best_search(family,v)
                update_family(family,best_match)
            for i in new_outlier:
                peaks_family.append({'mean':{'mz':i[0],'rt':i[1]},'peaks':[i],'id':len(peaks_family)+1})

    ###########-------------保存峰族---------------#############
    # 在保存前对元组进行标记
    def encode_tuple(obj):
        if isinstance(obj, tuple):
            return {'__tuple__': True, 'items': list(obj)}
        elif isinstance(obj, list):
            return [encode_tuple(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: encode_tuple(value) for key, value in obj.items()}
        return obj

    print(len(peaks_family))
    encoded_data = encode_tuple(peaks_family)

    # 保存到文件
    with open('peaks_family.json', 'w', encoding='utf-8') as f:
        json.dump(encoded_data, f, ensure_ascii=False, indent=4)

