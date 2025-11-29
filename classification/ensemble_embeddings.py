import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

import xgboost as xgb


def load_embeddings():
    w2v_df = pd.read_csv("data/word2vec_dataset.csv")
    s2v_df = pd.read_csv("data/sentence2vec_dataset.csv")
    d2v_df = pd.read_csv("data/doc2vec_dataset.csv")

    # they all share the same labels
    y = w2v_df["label"].values

    X_w2v = w2v_df.drop(columns=["label"])
    X_s2v = s2v_df.drop(columns=["label"])
    X_d2v = d2v_df.drop(columns=["label"])

    return X_w2v, X_s2v, X_d2v, y


def make_xgb():
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
    )


def train_base_xgb_models(X_w2v, X_s2v, X_d2v, y):
    """
    Single train/test split shared by all models.
    Returns trained models and test-set predictions.
    """
    n_samples = len(y)
    indices = np.arange(n_samples)

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    y_train = y[train_idx]
    y_test = y[test_idx]

    Xw_train, Xw_test = X_w2v.iloc[train_idx], X_w2v.iloc[test_idx]
    Xs_train, Xs_test = X_s2v.iloc[train_idx], X_s2v.iloc[test_idx]
    Xd_train, Xd_test = X_d2v.iloc[train_idx], X_d2v.iloc[test_idx]

    # train three XGB models
    print("\nTraining XGBoost on Word2Vec...")
    model_w2v = make_xgb()
    model_w2v.fit(Xw_train, y_train)

    print("Training XGBoost on Sentence2Vec...")
    model_s2v = make_xgb()
    model_s2v.fit(Xs_train, y_train)

    print("Training XGBoost on Doc2Vec...")
    model_d2v = make_xgb()
    model_d2v.fit(Xd_train, y_train)

    # individual metrics (for reference)
    for name, model, X_tr, X_te in [
        ("Word2Vec", model_w2v, Xw_train, Xw_test),
        ("Sentence2Vec", model_s2v, Xs_train, Xs_test),
        ("Doc2Vec", model_d2v, Xd_train, Xd_test),
    ]:
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        print(f"\nBase XGB — {name}")
        print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}")

    # predicted probs on test set (for ensembles)
    p_w2v_test = model_w2v.predict_proba(Xw_test)[:, 1]
    p_s2v_test = model_s2v.predict_proba(Xs_test)[:, 1]
    p_d2v_test = model_d2v.predict_proba(Xd_test)[:, 1]

    base_models = {
        "w2v": model_w2v,
        "s2v": model_s2v,
        "d2v": model_d2v,
    }

    test_data = {
        "Xw_train": Xw_train,
        "Xs_train": Xs_train,
        "Xd_train": Xd_train,
        "Xw_test": Xw_test,
        "Xs_test": Xs_test,
        "Xd_test": Xd_test,
        "y_train": y_train,
        "y_test": y_test,
        "p_w2v_test": p_w2v_test,
        "p_s2v_test": p_s2v_test,
        "p_d2v_test": p_d2v_test,
    }

    return base_models, test_data


def soft_voting_ensemble(test_data):
    """
    Simple soft voting: average the three probability predictions.
    """
    y_test = test_data["y_test"]
    p_w2v = test_data["p_w2v_test"]
    p_s2v = test_data["p_s2v_test"]
    p_d2v = test_data["p_d2v_test"]

    p_mean = (p_w2v + p_s2v + p_d2v) / 3.0

    y_pred = (p_mean >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, p_mean)

    print("\n==============================")
    print(" Soft Voting Ensemble (XGB probs)")
    print("==============================")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC–AUC:  {auc:.4f}")

    return acc, auc


def stacking_ensemble(X_w2v, X_s2v, X_d2v, y):
    """
    Proper stacking:
    - use K-fold cross-validation on the training portion
      to generate out-of-fold predictions from base models
    - train a Logistic Regression meta-learner on these
      predictions
    - evaluate on a held-out test set
    """

    n_samples = len(y)
    indices = np.arange(n_samples)

    # shared held-out test set
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    y_train = y[train_idx]
    y_test = y[test_idx]

    Xw_train, Xw_test = X_w2v.iloc[train_idx], X_w2v.iloc[test_idx]
    Xs_train, Xs_test = X_s2v.iloc[train_idx], X_s2v.iloc[test_idx]
    Xd_train, Xd_test = X_d2v.iloc[train_idx], X_d2v.iloc[test_idx]

    # out-of-fold predictions for meta-training
    oof_p_w2v = np.zeros(len(train_idx))
    oof_p_s2v = np.zeros(len(train_idx))
    oof_p_d2v = np.zeros(len(train_idx))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nGenerating out-of-fold predictions for stacking...")

    for fold, (tr, val) in enumerate(skf.split(Xw_train, y_train), start=1):
        print(f" Fold {fold}/5")

        Xw_tr, Xw_val = Xw_train.iloc[tr], Xw_train.iloc[val]
        Xs_tr, Xs_val = Xs_train.iloc[tr], Xs_train.iloc[val]
        Xd_tr, Xd_val = Xd_train.iloc[tr], Xd_train.iloc[val]
        y_tr, y_val = y_train[tr], y_train[val]

        # train base models on this fold
        mw = make_xgb()
        ms = make_xgb()
        md = make_xgb()

        mw.fit(Xw_tr, y_tr)
        ms.fit(Xs_tr, y_tr)
        md.fit(Xd_tr, y_tr)

        oof_p_w2v[val] = mw.predict_proba(Xw_val)[:, 1]
        oof_p_s2v[val] = ms.predict_proba(Xs_val)[:, 1]
        oof_p_d2v[val] = md.predict_proba(Xd_val)[:, 1]

    # meta-training set: predictions from base models
    meta_X_train = np.vstack([oof_p_w2v, oof_p_s2v, oof_p_d2v]).T
    meta_y_train = y_train

    # train meta-learner (logistic regression)
    meta_clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
    )
    meta_clf.fit(meta_X_train, meta_y_train)

    # for test meta-features, train base models on full train set
    mw_full = make_xgb()
    ms_full = make_xgb()
    md_full = make_xgb()

    mw_full.fit(Xw_train, y_train)
    ms_full.fit(Xs_train, y_train)
    md_full.fit(Xd_train, y_train)

    p_w2v_test = mw_full.predict_proba(Xw_test)[:, 1]
    p_s2v_test = ms_full.predict_proba(Xs_test)[:, 1]
    p_d2v_test = md_full.predict_proba(Xd_test)[:, 1]

    meta_X_test = np.vstack([p_w2v_test, p_s2v_test, p_d2v_test]).T

    p_meta = meta_clf.predict_proba(meta_X_test)[:, 1]
    y_pred = (p_meta >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, p_meta)

    print("\n==============================")
    print(" Stacking Ensemble (XGB base + LR meta)")
    print("==============================")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC–AUC:  {auc:.4f}")

    return acc, auc


if __name__ == "__main__":
    # load embeddings
    X_w2v, X_s2v, X_d2v, y = load_embeddings()

    # 1) Train base XGB models once and do soft-voting on their test predictions
    base_models, test_data = train_base_xgb_models(X_w2v, X_s2v, X_d2v, y)
    soft_acc, soft_auc = soft_voting_ensemble(test_data)

    # 2) Proper stacking with out-of-fold predictions and LR meta-learner
    stack_acc, stack_auc = stacking_ensemble(X_w2v, X_s2v, X_d2v, y)

    print("\n========= ENSEMBLE SUMMARY =========")
    print(f"Soft Voting: Acc={soft_acc:.4f}, AUC={soft_auc:.4f}")
    print(f"Stacking:    Acc={stack_acc:.4f}, AUC={stack_auc:.4f}")
