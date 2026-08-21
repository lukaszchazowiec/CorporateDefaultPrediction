import numpy as np
import pandas as pd

from cleaning import drop_missing

def altman_zscore(df):
    """
    Classic Altman z-score
    Z = 1,2X_1 + 1,4X_2 + 3,3X_3 + 0,6X_4 + 0,999X_5

    X1- Working Capital / Total Assets
    X2- Retained Earnings / Total Assets
    X3- EBIT / Total Assets
    X4- Book Value of Equity / Total Liabilities
    X5- Sales / Total Assets
    """

    z = (
            1.2 * df["A3"] +
            1.4 * df["A6"] +
            3.3 * df["A7"] +
            0.6 * df["A8"] +
            1.0 * df["A9"]
    )
    return z



if __name__ == "__main__":

    raw_df = pd.read_csv("../data/processed/year1.csv")
    df = drop_missing(raw_df)

    z_scores = altman_zscore(df)
    print(z_scores)