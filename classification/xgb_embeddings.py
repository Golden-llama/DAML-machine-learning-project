import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


def load_dataset(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y


def train_xgb(name, X, y):
    print(f"\n==============================")
    print(f" XGBoost — {name}")
    print(f"==============================")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model = xgb.XGBClassifier(
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

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy:  {acc:.4f}")
    print(f"ROC–AUC:   {auc:.4f}")

    return {
        "model": model,
        "accuracy": acc,
        "auc": auc,
    }


if __name__ == "__main__":
    datasets = {
        "Word2Vec": "data/word2vec_dataset.csv",
        "Sentence2Vec": "data/sentence2vec_dataset.csv",
        "Doc2Vec": "data/doc2vec_dataset.csv",
    }

    results = {}

    for name, path in datasets.items():
        print(f"\nLoading {name} from {path} ...")
        X, y = load_dataset(path)
        print(f"{name} shape: {X.shape}")
        print(f"{name} label counts:")
        print(y.value_counts())
        res = train_xgb(name, X, y)
        results[name] = res

    print("\nSUMMARY OF RESULTS (XGBoost)")
    for name, res in results.items():
        print(f"{name}: Acc={res['accuracy']:.4f}, AUC={res['auc']:.4f}")
