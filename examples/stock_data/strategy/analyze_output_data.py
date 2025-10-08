import os

import pandas as pd
import numpy as np
from empyrical import cum_returns


def analyze_output_data(file_path:str):
    df = pd.read_csv(file_path)
    df_score_ascending = df.sort_values(by='SCORE', ascending=False)
    print(df_score_ascending.head(10).iloc[:, 1:8])

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    analyze_output_data(f"{current_dir}/result/2025-09-30_output_sh.csv")
    analyze_output_data(f"{current_dir}/result/2025-09-30_output_sz.csv")

