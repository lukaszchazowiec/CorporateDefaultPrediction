import numpy as np
import pandas as pd


def compute_iv(df, column, n_bins=10):
    bins = pd.qcut(df[column], q=n_bins, duplicates="drop")

    grouped = df.groupby(bins, observed=True)["class"].agg(["count", "sum"])
    grouped.columns = ["total", "bad"]
    grouped["good"] = grouped["total"] - grouped["bad"]

    grouped["pct_good"] = grouped["good"] / grouped["good"].sum()
    grouped["pct_bad"] = grouped["bad"] / grouped["bad"].sum()

    grouped["woe"] = np.log(grouped["pct_good"] / grouped["pct_bad"])
    grouped["iv"] = (grouped["pct_good"] - grouped["pct_bad"]) * grouped["woe"]

    woe_series = bins.map(grouped["woe"]).astype(float).fillna(0)

    return grouped["iv"].sum(), woe_series


def woe_transform(df, column, woe_series_dict):

    return woe_series_dict[column]


def correlation_filter(df, sorted_cols, threshold=0.8):

    df_sorted = df[sorted_cols]

    corr_matrix = df_sorted.corr().abs()

    to_drop = set()

    for i in range(len(sorted_cols)):
        col_keep = sorted_cols[i]

        if col_keep in to_drop:
            continue

        for j in range(i + 1, len(sorted_cols)):
            col_eval = sorted_cols[j]

            if col_eval in to_drop:
                continue

            if corr_matrix.loc[col_keep, col_eval] > threshold:
                to_drop.add(col_eval)

    kept_cols = [c for c in sorted_cols if c not in to_drop]
    return kept_cols, df.drop(columns=list(to_drop))


if __name__ == "__main__":
    df = pd.read_csv("../data/processed/year1.csv")

    attr_cols = [c for c in df.columns if c.startswith("A")]

    iv_results = {}
    woe_series_dict = {}

    for col in attr_cols:
        try:
            iv, woe_col = compute_iv(df, col)
            iv_results[col] = iv
            woe_series_dict[col] = woe_col
        except Exception as e:
            print(f"Problem with {col}: {e}")

    iv_table = pd.Series(iv_results).sort_values(ascending=False)
    iv_table = iv_table[iv_table > 0.02]

    sorted_cols = list(iv_table.index)

    for col in sorted_cols:
        df[col] = woe_transform(df, col, woe_series_dict)

    print("Variables before the correlation filter (IV > 0.02):", len(sorted_cols))

    final_cols, df_final = correlation_filter(df, sorted_cols, threshold=0.8)

    print("Variables after the correlation filter (|rho| <= 0.8):", len(final_cols))
    print("Discarded variables:", set(sorted_cols) - set(final_cols))
    print("\nModel-ready data:")
    print(df_final[final_cols].head())