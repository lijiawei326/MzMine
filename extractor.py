import os
import pandas as pd
import numpy as np
from mzmine import mzmine,best_search


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



#######-------原始数据--------########
root = '../茅台质谱数据20240527-csv/基酒'
file_list = os.listdir(root)
data_dict = {f:pd.read_csv(os.path.join(root,f),index_col='retention_time') for f in file_list}
gcms_data = data_dict

for k,v in gcms_data.items():
    df = v.loc[:,'50':].copy()
    df[df>250000] = 250000
    df[df<1000] = 0
    gcms_data[k] = df


def load_peaks(file_name):
    data = np.load(os.path.join('./data_peaks',file_name))
    data = [tuple(i) for i in data]
    return data

data_peaks_files = os.listdir('./data_peaks')
data_peaks_files = {k.split('.')[0]+'.csv':load_peaks(k) for k in data_peaks_files}
gcms_data_peaks = data_peaks_files


#######-------峰数据--------########
def load_peaks(file_name):
    data = np.load(os.path.join('./data_peaks',file_name))
    data = [tuple(i) for i in data]
    return data

data_peaks_files = os.listdir('./data_peaks')
data_peaks_files = {k.split('.')[0]+'.csv':load_peaks(k) for k in data_peaks_files}
gcms_data_peaks = data_peaks_files


#######-------峰族--------########
# 读取并解码数据
import json

with open('peaks_family.json', 'r', encoding='utf-8') as f:
    loaded_data = json.load(f)

# 解码元组
def decode_tuple(obj):
    if isinstance(obj, dict) and '__tuple__' in obj and obj['__tuple__']:
        return tuple(obj['items'])
    elif isinstance(obj, list):
        return [decode_tuple(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: decode_tuple(value) for key, value in obj.items()}
    return obj

peaks_family = decode_tuple(loaded_data)


#######-------峰族--------########
df_data_peaks = pd.DataFrame(columns=range(1,len(peaks_family)+1))
for k,v in gcms_data.items():
    name = k
    df_feature = get_feature_peaks(peaks_family,name,v,gcms_data_peaks[k])
    df_data_peaks = pd.concat([df_data_peaks,df_feature])

df_data_peaks.to_csv('基酒特征.csv')



