from statistics import mean, stdev
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model, preprocessing
from sklearn.model_selection import StratifiedKFold, train_test_split
from imblearn.over_sampling import SMOTE

from cleaning import drop_missing

raw_df = pd.read_csv("../data/processed/year1.csv")
df = drop_missing(raw_df)

X = df.drop(columns=["class"])
y = df["class"]


def stratified_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_means = X_train.mean()
    X_train = X_train.fillna(train_means)
    X_test = X_test.fillna(train_means)

    strat_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    return X_train, X_test, y_train, y_test, strat_fold


X_train, X_test, y_train, y_test, strat_fold = stratified_split(X, y)

for fold, (train_idx, val_idx) in enumerate(
    strat_fold.split(X_train, y_train), 1
):
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]


def balance_classes(X_tr, y_tr, method):

    if method == "smote":
        smote = SMOTE(random_state=42   )
        X_resampled, y_resampled = smote.fit_resample(X_tr, y_tr)

        return X_resampled, y_resampled

    elif method == "class_weight":
        pass

    else:
        return "No such method found"