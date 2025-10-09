import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.time_features import time_features
import matplotlib.pyplot as plt

data_intervals = {  # train/val/test: [Custom] 7/1/2
    'ETTh1': [12*30*24, (12+4)*30*24, (12+8)*30*24],  # 12/4/4 months
    'ETTh2': [12*30*24, (12+4)*30*24, (12+8)*30*24],  # 12/4/4 months
    'Electricity': [24*30*24, (24+4)*30*24, (24+12)*30*24],  # 24/4/8 months
    'Weather': [8*30*24*6, int((8+4/3)*30*24*6), int((8+4)*30*24*6)],  # 8/(4/3)/(8/3) months
    'Exchange': [int(7588*0.7), int(7588*(0.7+0.1)), 7588],
    'Traffic': [16*30*24, (16+4)*30*24, (16+8)*30*24],  # 16/4/4 months
    'Appliance': [int(19735*0.7), int(19735*(0.7+0.1)), 19735],
    'Solar': [256*24*6, (256+35)*24*6, (256+109)*24*6]  # 256/35/74 days
}

class Dataset_MTS(Dataset):
    def __init__(self, data_name, data_path, cont_len, pred_len, status='train',
                 task='S', target='OT', scale=True, freq='h'):
        self.cont_len = cont_len
        self.pred_len = pred_len
        assert status in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[status]
        self.task = task
        self.target = target
        self.scale = scale
        self.freq = freq
        self.intervals = data_intervals[data_name]
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path)
        start_indices = [0, self.intervals[0]-self.cont_len, self.intervals[1]-self.cont_len]
        end_indices = [self.intervals[0], self.intervals[1], self.intervals[2]]
        start_index = start_indices[self.set_type]
        end_index = end_indices[self.set_type]

        if self.task == 'M' or self.task == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.task == 'S':
            df_data = df_raw[[self.target]]
        else:
            return NotImplementedError("No such variate!")

        if self.scale:
            train_data = df_data[start_indices[0]:end_indices[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][start_index:end_index]
        df_stamp['date'] = pd.to_datetime(df_stamp.date, format='mixed')
        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[start_index:end_index]
        self.data_y = data[start_index:end_index]
        self.data_stamp = data_stamp
        print(f"{self.data_path} parsing done!")

    def __getitem__(self, index):
        x_begin = index
        x_end = x_begin + self.cont_len
        y_begin = x_end
        y_end = y_begin + self.pred_len

        seq_x = self.data_x[x_begin:x_end]  # (B, cont_len, D)
        seq_y = self.data_y[y_begin:y_end]  # (B, pred_len, D)
        seq_x_mark = self.data_stamp[x_begin:x_end]  # (B, cont_len, 4)
        seq_y_mark = self.data_stamp[y_begin:y_end]  # (B, pred_len, 4)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.cont_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)