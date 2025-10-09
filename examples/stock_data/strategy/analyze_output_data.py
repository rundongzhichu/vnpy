import os

import pandas as pd
import numpy as np
from empyrical import cum_returns


def analyze_output_data(file_path:str):
    df = pd.read_csv(file_path)
    df_score_ascending = df.sort_values(by='SCORE', ascending=False)
    print(df_score_ascending.head(10).iloc[:, 1:8])
    df_score_ascending_filtered = df_score_ascending.loc[(df_score_ascending.NET_MAIN_INFLOW > -600).astype(bool) & (df_score_ascending.MAIN_FORCE_BUY > 600).astype(bool)]
    print(df_score_ascending_filtered.head(10).iloc[:, 1:8])

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    analyze_output_data(f"{current_dir}/result/2025-10-08_output_sh.csv")
    analyze_output_data(f"{current_dir}/result/2025-10-08_output_sz.csv")

