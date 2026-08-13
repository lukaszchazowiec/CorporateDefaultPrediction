import numpy as np
import pandas as pd


def compute_iv(df, column, n_bins=10):
    bins = pd.qcut(df[column], q=n_bins, duplicates="drop")

    grouped = df.groupby(bins)["class"].agg(["count", "sum"])
    grouped.columns = ["total", "bad"]
    grouped["good"] = grouped["total"] - grouped["bad"]

    grouped["pct_good"] = grouped["good"] / grouped["total"]
    grouped["pct_bad"] = grouped["bad"] / grouped["total"]

    grouped["woe"] = np.log(grouped["pct_good"] / grouped["pct_bad"])
    grouped["iv"] = (grouped["good"] - grouped["bad"]) * grouped["woe"]

    return grouped["iv"].sum()



if __name__ == "__main__":
    df = pd.read_csv("../data/processed/year1.csv")
    print(compute_iv(df, 'A1'))
