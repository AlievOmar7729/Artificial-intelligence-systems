import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

URL = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv"

def make_bins(price_arr, bins=2):
    y = pd.Series(price_arr).astype(float)
    if bins == 2:
        thr = y.median()
        lbl = (y > thr).astype(int)
    elif bins == 3:
        q1, q2 = y.quantile([0.33, 0.66])
        lbl = pd.cut(y, bins=[-np.inf, q1, q2, np.inf], labels=[0,1,2]).astype(int)
    else:
        raise ValueError("bins must be 2 or 3")
    return lbl.values

def clean_data(df):
    df = df.copy()
    if "price" not in df.columns:
        raise ValueError(f"Не знайдено 'price'. Є тільки: {df.columns.tolist()}")
    df = df.dropna(subset=["price"]).drop_duplicates()
    df = df[df["price"] > 0]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df

def make_feature_matrix(df, num_cols, cat_cols, one_hot=True):
    Xnum = df[num_cols].astype(float).values if num_cols else np.zeros((len(df), 0))
    Xcat = None

    if cat_cols:
        if one_hot:
            oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            Xcat = oh.fit_transform(df[cat_cols])
        else:
            mats = []
            for c in cat_cols:
                le = LabelEncoder()
                mats.append(le.fit_transform(df[c]).reshape(-1, 1))
            Xcat = np.hstack(mats)

    if Xcat is None:
        return Xnum
    if Xnum.size == 0:
        return Xcat
    return np.hstack([Xnum, Xcat])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=2, choices=[2,3], help="бінування ціни")
    ap.add_argument("--model", type=str, default="gauss", choices=["gauss","multi"])
    ap.add_argument("--scale", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(URL)
    df = clean_data(df)

    keep_cols = ["origin","destination","train_type","train_class","fare",
                 "duration","distance","price","insert_date","start_date"]
    df = df[[c for c in keep_cols if c in df.columns]].dropna()

    y = make_bins(df["price"].values, bins=args.bins)

    num_cols = [c for c in ["duration","distance"] if c in df.columns]
    cat_cols = [c for c in ["origin","destination","train_type","train_class","fare"] if c in df.columns]

    X = make_feature_matrix(df, num_cols, cat_cols, one_hot=True)

    if args.scale and num_cols:
        sc = StandardScaler()
        X[:, :len(num_cols)] = sc.fit_transform(X[:, :len(num_cols)])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

    clf = GaussianNB() if args.model == "gauss" else MultinomialNB()
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)

    print("Confusion matrix:\n", confusion_matrix(yte, yp))
    print("\nClassification report:\n", classification_report(yte, yp, digits=4))

if __name__ == "__main__":
    main()
