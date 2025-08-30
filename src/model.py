#!/usr/bin/env python3
# src/model.py
import argparse
import json
from pathlib import Path
from features import assign_label, prompt_classify


import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# We reuse your labeling helpers from features.py
from src.features import assign_label, prompt_classify  # type: ignore


def make_labels(df: pd.DataFrame, ambiguous_strategy: str = "valid") -> pd.DataFrame:
    """
    Create a 'label' column using heuristics first, then resolve ambiguous ones per strategy:

    ambiguous_strategy:
      - 'valid':   set ambiguous to 'Valid' (default).
      - 'drop':    drop ambiguous rows from training set.
      - 'llm':     call prompt_classify(text_clean) to label ambiguous rows via LLM.
    """
    # Heuristic labels first
    df = df.copy()
    df["label"] = df.apply(assign_label, axis=1)

    mask_amb = df["label"].isna()
    if mask_amb.any():
        if ambiguous_strategy == "valid":
            df.loc[mask_amb, "label"] = "Valid"
        elif ambiguous_strategy == "drop":
            df = df[~mask_amb].copy()
        elif ambiguous_strategy == "llm":
            # This will use your text-generation pipeline (Qwen) from features.py
            df.loc[mask_amb, "label"] = df.loc[mask_amb, "text_clean"].apply(prompt_classify)
        else:
            raise ValueError(f"Unknown ambiguous_strategy: {ambiguous_strategy}")

    # Final tidy
    df["label"] = df["label"].astype(str)
    return df


def build_label_maps(labels: pd.Series):
    labels_cat = labels.astype("category")
    id2label = {int(i): lab for i, lab in enumerate(labels_cat.cat.categories)}
    label2id = {lab: i for i, lab in id2label.items()}
    return label2id, id2label


def train_tfidf_logreg(
    texts: pd.Series,
    labels: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    pipe = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            strip_accents="unicode"
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="saga",
            n_jobs=-1,
            verbose=0
        )),
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)

    acc = accuracy_score(y_val, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, preds, average="macro", zero_division=0)
    report = classification_report(y_val, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_val, preds, labels=sorted(np.unique(labels)))

    metrics = {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "support": int(len(y_val)),
    }
    return pipe, (X_val, y_val, preds, report, cm, metrics)


def save_confusion_matrix(cm: np.ndarray, classes: list[str], out_png: Path):
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="Input parquet from preprocess.py")
    ap.add_argument("--out_dir", required=True, help="Directory to save model + artifacts")
    ap.add_argument("--ambiguous_strategy", choices=["valid", "drop", "llm"], default="valid",
                    help="How to label ambiguous (non-heuristic) cases")
    ap.add_argument("--test_size", type=float, default=0.2, help="Validation split size (0-1)")
    args = ap.parse_args()

    parquet_path = Path(args.parquet)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load cleaned reviews
    df = pd.read_parquet(parquet_path)

    # Sanity checks for required columns
    needed_cols = {"text_clean", "looks_promo_heur", "looks_rant_no_visit_heur"}
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for labeling: {missing}. "
                         f"Run preprocess to produce heuristics before modeling.")

    # Build labels using heuristics (+ optional LLM for ambiguous)
    df = make_labels(df, ambiguous_strategy=args.ambiguous_strategy)

    # Build label maps
    label2id, id2label = build_label_maps(df["label"])

    # Train TF-IDF + Logistic Regression
    model, (X_val, y_val, preds, report, cm, metrics) = train_tfidf_logreg(
        df["text_clean"].fillna(""),
        df["label"].map(label2id)
    )

    # Persist artifacts
    joblib.dump(model, out_dir / "tfidf_logreg.joblib")
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2, ensure_ascii=False)

    # Save metrics
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save detailed report (per-class)
    rep_df = pd.DataFrame(report).T
    rep_df.to_csv(out_dir / "classification_report.csv")

    # Confusion matrix (use class order by id2label index)
    class_ids = sorted(id2label.keys())
    class_names = [id2label[i] for i in class_ids]
    save_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")

    # Also save a small preview CSV for manual inspection
    prev = pd.DataFrame({
        "text_clean": X_val,
        "true": [id2label[int(y)] for y in y_val],
        "pred": [id2label[int(p)] for p in preds],
    })
    prev.head(200).to_csv(out_dir / "val_preview_sample.csv", index=False)

    print("Saved model + artifacts to:", str(out_dir))
    print("Metrics (macro):", metrics)
    print("Classes:", id2label)


if __name__ == "__main__":
    main()
