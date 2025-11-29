import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


def train_and_eval(name, X, y):
    print(f"\n==============================")
    print(f" Logistic Regression — {name}")
    print(f"==============================")

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # scale dense embeddings (helps LR a lot)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy:  {acc:.4f}")
    print(f"ROC–AUC:   {auc:.4f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    return acc, auc


if __name__ == "__main__":

    # ===== 1) Word2Vec =====
    w2v_df = pd.read_csv("data/word2vec_dataset.csv")
    X_w2v = w2v_df.drop(columns=["label"])
    y_w2v = w2v_df["label"]

    print("Word2Vec shape:", X_w2v.shape)
    print("Word2Vec label counts:")
    print(y_w2v.value_counts())

    acc_w2v, auc_w2v = train_and_eval("Word2Vec", X_w2v, y_w2v)

    # ===== 2) Sentence2Vec =====
    s2v_df = pd.read_csv("data/sentence2vec_dataset.csv")

    # in case text columns are still in there, only keep s2v_* columns
    s2v_feature_cols = [c for c in s2v_df.columns if c.startswith("s2v_")]
    X_s2v = s2v_df[s2v_feature_cols]
    y_s2v = s2v_df["label"]

    print("\nSentence2Vec shape:", X_s2v.shape)
    print("Sentence2Vec label counts:")
    print(y_s2v.value_counts())

    acc_s2v, auc_s2v = train_and_eval("Sentence2Vec", X_s2v, y_s2v)

    # ===== 3) Doc2Vec =====
    d2v_df = pd.read_csv("data/doc2vec_dataset.csv")
    X_d2v = d2v_df.drop(columns=["label"])
    y_d2v = d2v_df["label"]

    print("\nDoc2Vec shape:", X_d2v.shape)
    print("Doc2Vec label counts:")
    print(y_d2v.value_counts())

    acc_d2v, auc_d2v = train_and_eval("Doc2Vec", X_d2v, y_d2v)

    # ===== Summary =====
    print("\n========= SUMMARY (Logistic Regression on embeddings) =========")
    print(f"Word2Vec:     Acc={acc_w2v:.4f}, AUC={auc_w2v:.4f}")
    print(f"Sentence2Vec: Acc={acc_s2v:.4f}, AUC={auc_s2v:.4f}")
    print(f"Doc2Vec:      Acc={acc_d2v:.4f}, AUC={auc_d2v:.4f}")
