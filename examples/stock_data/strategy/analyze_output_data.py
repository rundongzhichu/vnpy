import pandas as pd
import numpy as np


def analyze_output_data(file_path:str):
    df = pd.read_csv(file_path)
    df = df[(df["大单买入量"] > 1000) & (df["主力净流入总额"] > -2000)]
    print(df.iloc[:, 1:8])

if __name__ == "__main__":
    analyze_output_data("../demo/output_sh.csv")
    analyze_output_data("../demo/output_sz.csv")

