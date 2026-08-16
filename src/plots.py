import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from woe_iv import compute_iv


def plot_woe(df, top_n=5):
    attr_cols = [c for c in df.columns if c.startswith("A")]
    iv_results = {}
    grouped_details = {}

    for col in attr_cols:
        try:
            iv, woe_col = compute_iv(df, col)
            iv_results[col] = iv
        except Exception as e:
            print(f"Problem with {col}: {e}")

    iv_table = pd.Series(iv_results).sort_values(ascending=False)
    iv_table = iv_table[iv_table > 0.02]
    print("IV table (top):")
    print(iv_table.head(top_n))

    for col in iv_table.head(top_n).index:
        bins = pd.qcut(df[col], q=10, duplicates="drop")
        grouped = (
            df.groupby(bins, observed=True)["class"]
            .agg(["count", "sum"])
            .rename(columns={"sum": "bad"})
        )
        grouped["good"] = grouped["count"] - grouped["bad"]

        grouped["pct_good"] = grouped["good"] / grouped["good"].sum()
        grouped["pct_bad"] = grouped["bad"] / grouped["bad"].sum()
        grouped["woe"] = np.log(grouped["pct_good"] / grouped["pct_bad"])

        plt.figure(figsize=(10, 4))
        plt.bar(range(len(grouped)), grouped["woe"], color="skyblue")
        plt.axhline(0, color="red", linestyle="--")
        plt.xticks(
            range(len(grouped)),
            [str(b) for b in grouped.index],
            rotation=45,
            ha="right",
        )
        plt.title(f"WoE profile for {col} (IV = {iv_table[col]:.4f})")
        plt.xlabel("Bins")
        plt.ylabel("WoE")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    df_data = pd.read_csv("../data/processed/year1.csv")
    plot_woe(df_data, top_n=3)