import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split


def compute_iv(df, column, n_bins=5):

    bins, bin_edges = pd.qcut(
        df[column], q=n_bins, duplicates="drop", retbins=True
    )

    grouped = df.groupby(bins, observed=True)["class"].agg(["count", "sum"])
    grouped.columns = ["total", "bad"]
    grouped["good"] = grouped["total"] - grouped["bad"]

    epsilon = 1e-5
    grouped["pct_good"] = (grouped["good"] + epsilon) / (
        grouped["good"].sum() + epsilon
    )
    grouped["pct_bad"] = (grouped["bad"] + epsilon) / (
        grouped["bad"].sum() + epsilon
    )

    grouped["woe"] = np.log(grouped["pct_good"] / grouped["pct_bad"])
    grouped["iv"] = (grouped["pct_good"] - grouped["pct_bad"]) * grouped["woe"]

    total_iv = grouped["iv"].sum()

    return total_iv, bin_edges, grouped


def check_monotonicity(woe_values):

    is_increasing = all(
        x <= y for x, y in zip(woe_values[:-1], woe_values[1:])
    )
    is_decreasing = all(
        x >= y for x, y in zip(woe_values[:-1], woe_values[1:])
    )
    return is_increasing or is_decreasing


def woe_transform(df, column, bin_edges, grouped_info):

    adjusted_edges = bin_edges.copy()
    adjusted_edges[0] = -np.inf
    adjusted_edges[-1] = np.inf

    binned = pd.cut(df[column], bins=adjusted_edges)

    woe_map = grouped_info["woe"].to_dict()
    woe_series = binned.map(woe_map).astype(float)

    return woe_series


def plot_woe(column, grouped_info, iv_value):

    woe_values = grouped_info["woe"].values
    is_monotonic = check_monotonicity(woe_values)

    plt.figure(figsize=(9, 4))
    bars = plt.bar(
        range(len(grouped_info)),
        grouped_info["woe"],
        color="skyblue" if is_monotonic else "orange",
    )

    plt.axhline(0, color="red", linestyle="--")
    plt.xticks(
        range(len(grouped_info)),
        [str(b) for b in grouped_info.index],
        rotation=30,
        ha="right",
    )

    status_str = "Monotonic" if is_monotonic else "Non-Monotonic (Review)"
    plt.title(
        f"WoE Profile: {column} | IV = {iv_value:.4f} | Status: {status_str}"
    )
    plt.xlabel("Bins (Quintiles)")
    plt.ylabel("Weight of Evidence (WoE)")
    plt.tight_layout()
    plt.show()


def correlation_filter(df_woe, sorted_cols, threshold=0.8):

    corr_matrix = df_woe[sorted_cols].corr().abs()
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
    return kept_cols


if __name__ == "__main__":
    df = pd.read_csv("../data/processed/year1.csv")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["class"]
    )

    attr_cols = [c for c in train_df.columns if c.startswith("A")]

    iv_results = {}
    bin_edges_dict = {}
    grouped_info_dict = {}

    for col in attr_cols:
        try:
            iv, edges, grouped = compute_iv(train_df, col, n_bins=5)
            iv_results[col] = iv
            bin_edges_dict[col] = edges
            grouped_info_dict[col] = grouped
        except Exception as e:
            print(f"Skipping {col} due to binning error: {e}")

    iv_table = pd.Series(iv_results).sort_values(ascending=False)
    iv_table = iv_table[iv_table >= 0.02]
    sorted_cols = list(iv_table.index)

    print(
        f"Selected {len(sorted_cols)} features with IV >= 0.02 out of {len(attr_cols)}."
    )

    top_features = sorted_cols[:3]
    for col in top_features:
        plot_woe(col, grouped_info_dict[col], iv_table[col])

    train_woe = pd.DataFrame(index=train_df.index)
    test_woe = pd.DataFrame(index=test_df.index)

    for col in sorted_cols:
        train_woe[col] = woe_transform(
            train_df, col, bin_edges_dict[col], grouped_info_dict[col]
        )
        test_woe[col] = woe_transform(
            test_df, col, bin_edges_dict[col], grouped_info_dict[col]
        )

    final_cols = correlation_filter(train_woe, sorted_cols, threshold=0.8)

    train_woe_final = train_woe[final_cols]
    test_woe_final = test_woe[final_cols]

    print(f"Final selected features after correlation filter: {len(final_cols)}")
    print("Features ready for logistic regression:", final_cols)