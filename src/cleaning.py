import pandas as pd
from pandas import isnull

df = pd.read_csv("../data/processed/year1.csv")
print(df.info())

df = df.loc[:, df.isnull().mean() < 0.3]
print(df.columns)

data = df.fillna(df.mean())
print(data.head())
print(data.info())