import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from cleaning import drop_missing

def altman_zscore(df):
    """
    Altman Z'-score (1983) - variant for privately-held manufacturing firms,
    using book value of equity instead of market value.

    X1 = Working Capital / Total Assets       -> A3
    X2 = Retained Earnings / Total Assets     -> A6
    X3 = EBIT / Total Assets                  -> A7
    X4 = Book Value of Equity / Total Liab.   -> A8
    X5 = Sales / Total Assets                 -> A9
    """
    z_score = (
        0.717 * df["A3"] +
        0.847 * df["A6"] +
        3.107 * df["A7"] +
        0.420 * df["A8"] +
        0.998 * df["A9"]
    )
    return z_score


def benchmark_auc(y_true, z_scores):
    risk_scores = -z_scores

    auc = roc_auc_score(y_true, risk_scores)
    gini = 2 * auc - 1

    return auc, gini


if __name__ == "__main__":

    raw_df = pd.read_csv("../data/processed/year1.csv")
    df = drop_missing(raw_df)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["class"]
    )

    test_z_scores = altman_zscore(test_df)
    auc, gini = benchmark_auc(test_df["class"], test_z_scores)

    print("Altman Z'-score benchmark")
    print(f"ROC-AUC : {auc:.4f}")
    print(f"Gini    : {gini:.4f}")