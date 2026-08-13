import numpy as np
import pandas as pd

df = pd.read_csv("../data/processed/year1.csv")

def drop_missing(dataframe):
    df = dataframe.loc[:, dataframe.isnull().mean() < 0.3]
    data = df.fillna(df.mean())

    return data

