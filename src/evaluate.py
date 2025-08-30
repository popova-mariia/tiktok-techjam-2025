# src/evaluate.py
"""
Evaluate review moderation on a cleaned parquet.

Modes:
  heuristic : ORs selected heuristic flags to produce a binary prediction
  sk        : uses your scikit-learn TF-IDF+LogReg model (from src/model.py)

Ground-truth:
  --label_col defaults to 'policy_violation_any' (binary 0/1)
  If you trained multi-class in model.py and saved a labeled parquet,
  set --label_col to 'label' and provide --sk_label_map/--sk_positive_label
  to convert to binary for metric reporting if needed.

Examples:
  # Heuristic-only vs policy_violation_any
  python src/evaluate.py --input data/processed/wyoming_reviews_clean.parquet --mode heuristic

  # Scikit-learn model vs labeled parquet saved by model.py
  python src/evaluate.py --input models/labeled_reviews.parquet --mode sk \
      --sk_model models/tfidf_logreg.joblib --sk_label_map models/label_map.json \
      --label_col label --sk_binary --sk_positive_label Violation
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    classification_report, confusion_matrix
)

DEFAULT_HEUR_FLAGS = ["policy_ads", "policy_irrelevant", "policy_novisit", "policy_low_quality"]

# -------- Heuristic predictions --------
def heuristic_predict(df: pd.DataFrame, flags: list[str]) -> np.ndarray:
    pred = np.zeros(len(df), dtype=bool)
    for f in flags:
        if f not in df.columns:
            raise SystemExit(f"Missing heuristic flag column: {f}")
        pred |= df[f].fillna(False).astype(bool).to_numpy()
    return pred.astype(int)

# -------- SK helpers --------
def sk_load_model(model_path: str):
    import joblib
    return joblib.load(model_path)

def sk_predict_texts(sk_model, texts: list[str]) -> np.ndarray:
    return sk_model.predict(texts)

def load_label_map(label_map_path: str) -> dict:
    lm = json.loads(Path(label_map_path).read_text(encoding="utf-8"))
    if "id2label" in lm:
        id2label = {int(k): v for k, v in lm["id2label"].items()}
        label2id = {v: k for k, v in id2label.items()}
    elif "label2id" in lm:
        label2id = lm["label2id"]
        id2label = {int(v): k for k, v in label2id.items()}
    else:
        raise SystemExit("label_map.json must contain id2label or label2id.")
    return {"id2label": id2label, "label2id": label2id}

def map_multiclass_to_binary(pred_ids: np.ndarray, id2label: dict, positive_label: str) -> np.ndarray:
    pos = positive_label.strip().lower()
    out = []
    for pid in pred_ids:
        lab = id2label[int(pid)]
        out.append(1 if lab.lower() == pos else 0)
    return np.array(out, dtype=int)

# -------- Metrics / report --------
def evaluate_and_report(y_true, y_pred, out_dir: Path, prefix: str = ""):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # choose averaging
    average = "binary" if set(np.unique(y_true)).issubset({0,1}) else "macro"

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec  = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1   = f1_score(y_true, y_pred, average=average, zero_division=0)

    print(f"\n=== {prefix}RESULTS ===")
    print(f"accuracy:  {acc:.4f}")
    print(f"precision: {prec:.4f}")
    print(f"recall:    {rec:.4f}")
    print(f"f1:        {f1:.4f}\n")

    # detailed report
    rep_txt = classification_report(y_true, y_pred, zero_division=0)
    print(rep_txt)

    out_dir.mkdir(parents=True, exist_ok=True)
    # save text report
    (out_dir / f"{prefix}classification_report.txt").write_text(rep_txt, encoding="utf-8")

    # confusion matrix
    labels_sorted = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted).to_csv(out_dir / f"{prefix}confusion_matrix.csv")

    # try to save a PNG (no seaborn; plain matplotlib)
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6,5))
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"Confusion Matrix ({prefix.strip().upper()})")
        plt.colorbar()
        ticks = np.arange(len(labels_sorted))
        plt.xticks(ticks, labels_sorted, rotation=45, ha="right")
        plt.yticks(ticks, labels_sorted)
        thresh = cm.max()/2 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], "d"),
                         ha="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.ylabel("True")
        plt.xlabel("Pred")
        plt.tight_layout()
        fig.savefig(out_dir / f"{prefix}confusion_matrix.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass

    # summary json
    summary = dict(accuracy=acc, precision=prec, recall=rec, f1=f1, labels=[int(x) for x in labels_sorted])
    (out_dir / f"{prefix}metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

# -------- Main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Parquet path (cleaned or labeled)")
    ap.add_argument("--mode", required=True, choices=["heuristic", "sk"])
    ap.add_argument("--out_dir", default="eval_out", help="Where to save metrics/artifacts")

    # data columns
    ap.add_argument("--label_col", default="policy_violation_any", help="Ground truth column")
    ap.add_argument("--text_col", default="text_clean", help="Text column for SK model")

    # heuristics
    ap.add_argument("--heur_flags", nargs="*", default=DEFAULT_HEUR_FLAGS, help="Heuristic columns to OR together")

    # sklearn
    ap.add_argument("--sk_model", help="Path to joblib model (required for mode=sk)")
    ap.add_argument("--sk_label_map", help="label_map.json (if converting multiclass → binary)")
    ap.add_argument("--sk_binary", action="store_true", help="Convert SK predictions to binary with --sk_positive_label")
    ap.add_argument("--sk_positive_label", default=None, help="Label string to treat as positive when --sk_binary")

    args = ap.parse_args()

    df = pd.read_parquet(args.input)

    if args.label_col not in df.columns:
        raise SystemExit(f"Label column '{args.label_col}' not found in {args.input}")

    y_true = df[args.label_col]
    # normalize to 0/1 if it's boolean
    if y_true.dtype == bool or set(y_true.dropna().astype(str).unique()).issubset({"True","False"}):
        y_true = y_true.astype(str).map({"True":1, "False":0}).astype(int)

    # Heuristic predictions (always useful to compare)
    y_pred_heur = heuristic_predict(df, args.heur_flags)

    if args.mode == "heuristic":
        evaluate_and_report(y_true, y_pred_heur, Path(args.out_dir), prefix="heur_")
        # also save per-row csv
        pd.DataFrame({"text": df.get("text_clean",""), "y_true": y_true, "y_pred": y_pred_heur}).to_csv(
            Path(args.out_dir) / "heur_predictions.csv", index=False
        )
        return

    # mode == sk
    if not args.sk_model:
        raise SystemExit("--sk_model is required for mode=sk")

    skm = sk_load_model(args.sk_model)
    texts = df[args.text_col].fillna("").astype(str).tolist()
    sk_preds_raw = sk_predict_texts(skm, texts)

    if args.sk_binary:
        if not args.sk_label_map or not args.sk_positive_label:
            raise SystemExit("--sk_label_map and --sk_positive_label are required when using --sk_binary")
        lm = load_label_map(args.sk_label_map)
        y_pred_sk = map_multiclass_to_binary(sk_preds_raw, lm["id2label"], args.sk_positive_label)
    else:
        # assume model already outputs {0,1}
        y_pred_sk = sk_preds_raw.astype(int)

    # evaluate both for comparison
    evaluate_and_report(y_true, y_pred_heur, Path(args.out_dir), prefix="heur_")
    evaluate_and_report(y_true, y_pred_sk, Path(args.out_dir), prefix="sk_")

    # save per-row predictions
    pd.DataFrame({
        "text": texts,
        "y_true": y_true,
        "heur_pred": y_pred_heur,
        "sk_pred": y_pred_sk
    }).to_csv(Path(args.out_dir) / "sk_predictions.csv", index=False)

if __name__ == "__main__":
    main()
