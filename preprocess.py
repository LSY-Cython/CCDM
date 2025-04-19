import pandas
import os
import numpy as np
import pandas as pd

def solar_txt2csv(input_file):
    df_raw = []
    with open(input_file, "r", encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n').split(',')
            data_line = np.stack([float(i) for i in line])
            df_raw.append(data_line)
    df_raw = np.stack(df_raw, 0)
    headers = [f"PV{i}" for i in range(137)]
    df_raw = pd.DataFrame(df_raw, columns=headers)
    df_raw.to_csv("dataset/Solar/solar.csv")

if __name__ == "__main__":
    solar_txt2csv("dataset/Solar/solar_AL.txt")