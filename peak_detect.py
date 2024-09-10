import numpy as np
import pandas as pd
import os
from findpeaks import findpeaks


def get_peak(df, denoise=None, method = 'topology',window=7):
    fp = findpeaks(method=method,whitelist='peak',denoise=denoise,lookahead=1,limit=-np.inf,params={'window':window})
    res = fp.fit(df.T.values)
    if method == 'topology':
        persistence = res['persistence']
        persistence = persistence[persistence['score'] > 255*(1000/250000)]
        peak_df = persistence[persistence['peak']].loc[:,['x','y','score']]
        peak_df.columns = ['rt','mz','score']
    else:
        mask = res['Xdetect']
        peak_list = []
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i,j]:
                    peak_list.append([i,j])
        peak_df = pd.DataFrame(peak_list,columns=['mz','rt'])
    return peak_df


def get_low_peak(df,high_peak_num):
    keep_indices = set(df.index[:high_peak_num])
    low_indices = set()
    # df = df.sort_values(by='score', ascending=False)
    # 遍历每一行
    for index, row in df.iloc[high_peak_num:].iterrows():
        # 检查当前行是否在任何已保留行的窗口内
        is_in_window = False
        for keep_index in keep_indices:
            if len(df.loc[[keep_index]]) > 1:
                print('index:',keep_index)
                print('rt:',df.loc[keep_index, 'rt'])
                print('mz',df.loc[keep_index, 'mz'])
            if abs(df.loc[keep_index, 'rt'] - row['rt']) <= 20 and abs(df.loc[keep_index, 'mz'] - row['mz']) <= 1:
                is_in_window = True
                break
        
        # 如果不在任何保留行的窗口内，则保留该行
        if not is_in_window:
            keep_indices.add(index)
            low_indices.add(index)

    # 使用保留的行索引筛选 DataFrame
    df_filtered = df.loc[list(low_indices)]
    return df_filtered


def get_all_peaks(df1):
    mzs = []
    rts = []

    # 去噪
    ids = []
    peak_df = get_peak(df1,denoise='mean',window=3,method='topology')
    peak_df = peak_df[peak_df['score'] >= 255*(3000/250000)]
    for i in peak_df.index:
        rt = df1.index[peak_df.loc[i,'rt']]
        mz = df1.columns[peak_df.loc[i,'mz']]
        peak_intensity = df1.loc[rt,mz]
        if peak_intensity > 1000:
            ids.append(i)
    peak_df = peak_df.loc[ids]
    mz_list = [int(df1.columns[peak_df.loc[i,'mz']]) for i in peak_df.index]
    rt_list = [df1.index[peak_df.loc[i,'rt']] for i in peak_df.index]
    mzs.extend(mz_list)
    rts.extend(rt_list)

    # 非去噪
    peak_df_new = get_peak(df1,denoise=None)
    ids = []
    candidate = peak_df_new[peak_df_new['score'] < 255*(4000/250000)]
    all = pd.concat([peak_df,candidate],ignore_index=True).drop_duplicates(keep='first')
    high_peak_num = len(peak_df)
    peak_df2 = get_low_peak(all,high_peak_num)
    for i in peak_df2.index:
        rt = df1.index[peak_df2.loc[i,'rt']]
        mz = df1.columns[peak_df2.loc[i,'mz']]
        peak_intensity = df1.loc[rt,mz]
        if peak_intensity > 1000:
            ids.append(i)
    peak_df2 = peak_df2.loc[ids]
    mz_list = [int(df1.columns[peak_df2.loc[i,'mz']]) for i in peak_df2.index]
    rt_list = [df1.index[peak_df2.loc[i,'rt']] for i in peak_df2.index]
    mzs.extend(mz_list)
    rts.extend(rt_list)
    return [(mz,rt) for mz,rt in zip(mzs,rts)]




root = '../茅台质谱数据20240527-csv/基酒'
file_list = os.listdir(root)
data_dict = {f:pd.read_csv(os.path.join(root,f),index_col='retention_time') for f in file_list}
gcms_data = data_dict

# 数据预处理
for k,v in gcms_data.items():
    df = v.loc[:,'50':].copy()
    df[df>250000] = 250000
    df[df<1000] = 0
    gcms_data[k] = df


gcms_data_peaks = {k:get_all_peaks(v) for k,v in gcms_data.items()}
for k,v in gcms_data_peaks.items():
    name = k.split('.')[0]
    np.save(f'./data_peaks/{name}.npy', np.array(v))