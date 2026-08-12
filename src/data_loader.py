import pandas as pd
from pathlib import Path

# Choose year to work with (1, 2, 3, 4, or 5)
YEAR = 1

RAW_DATA_PATH = Path("../data/raw/data.csv")
OUTPUT_PATH = Path(f"../data/processed/year{YEAR}.csv")

data = pd.read_csv(RAW_DATA_PATH)

data = data[data["year"] == YEAR]

n_companies = len(data)
n_bankrupt = data["class"].sum()

print("Year:", YEAR)
print("Number of companies:", n_companies)
print("Number of bankruptcies:", n_bankrupt)
print("Bankruptcy rate: {:.1f}%".format(n_bankrupt / n_companies * 100))

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
data.to_csv(OUTPUT_PATH, index=False)
print("Saved to:", OUTPUT_PATH)