import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "g_mikadze2024_46829.csv")
OUT_DIR = os.path.join(HERE, "outputs")

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)

# Keep this list simple and explain it in README
SPAM_WORDS = {
    "free", "winner", "win", "prize", "cash", "urgent", "offer", "limited",
    "buy", "cheap", "discount", "click", "claim", "bonus", "money", "deal",
    "guarantee", "congratulations", "loan", "credit", "exclusive", "investment", "now"
}

FEATURES = ["words", "links", "capital_words", "spam_word_count"]
TARGET = "is_spam"

def extract_features_from_text(text: str) -> pd.DataFrame:
    words = re.findall(r"[A-Za-z']+", text)
    word_count = len(words)
    links = len(URL_RE.findall(text))
    capital_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    spam_word_count = sum(1 for w in words if w.lower() in SPAM_WORDS)

    return pd.DataFrame([{
        "words": word_count,
        "links": links,
        "capital_words": capital_words,
        "spam_word_count": spam_word_count
    }])

def plot_class_distribution(y: pd.Series, out_path: str):
    counts = y.value_counts().sort_index()
    plt.figure()
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Class Distribution (0=Not Spam, 1=Spam)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_confusion_matrix(cm: np.ndarray, out_path: str):
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_coefficients(model: LogisticRegression, feature_names: list, out_path: str):
    coefs = model.coef_[0]
    plt.figure()
    plt.bar(feature_names, coefs)
    plt.title("Logistic Regression Feature Coefficients")
    plt.xlabel("Feature")
    plt.ylabel("Coefficient")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    # plots based on full dataset
    plot_class_distribution(y, os.path.join(OUT_DIR, "class_distribution.png"))

    # train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    plot_confusion_matrix(cm, os.path.join(OUT_DIR, "confusion_matrix.png"))
    plot_coefficients(model, FEATURES, os.path.join(OUT_DIR, "feature_coefficients.png"))

    print("=== Task 2: Spam Detector (Logistic Regression) ===")
    print("Features:", FEATURES)
    print("Intercept:", float(model.intercept_[0]))
    print("Coefficients:", dict(zip(FEATURES, model.coef_[0])))
    print("Confusion matrix:\n", cm)
    print(f"Accuracy: {acc:.4f}")
    print(f"Saved plots -> {OUT_DIR}")

    # interactive prediction
    print("\n--- Try your own email text ---")
    text = input("Paste an email text and press Enter:\n> ").strip()
    if text:
        feats = extract_features_from_text(text)
        proba = model.predict_proba(feats)[0][1]
        pred = int(model.predict(feats)[0])
        label = "SPAM (1)" if pred == 1 else "NOT SPAM (0)"
        print("\nExtracted features:", feats.to_dict(orient="records")[0])
        print(f"Prediction: {label}")
        print(f"Spam probability: {proba:.4f}")
    else:
        print("No input provided. Done.")

if __name__ == "__main__":
    main()
