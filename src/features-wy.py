# src/features-wy.py
import pandas as pd
from typing import Dict, Tuple, List, Optional

# Optional HF bits (loaded lazily only if use_llm=True or you pass a model_name)
try:
    from transformers import AutoTokenizer, pipeline
except Exception:
    AutoTokenizer = None
    pipeline = None


# ---------- Labeling logic ----------
# Priority order ensures exactly ONE label per row.
PRIORITY = ["Advertisement", "Irrelevant", "Rant_No_Visit", "Low_Quality", "Valid"]

def assign_label_from_policies(row: pd.Series) -> str:
    """
    Map new preprocessor flags to a single label.
    Columns expected (new dataset): policy_ads, policy_irrelevant, policy_novisit, policy_low_quality
    """
    if bool(row.get("policy_ads", False)):
        return "Advertisement"
    if bool(row.get("policy_irrelevant", False)):
        return "Irrelevant"
    if bool(row.get("policy_novisit", False)):
        return "Rant_No_Visit"
    if bool(row.get("policy_low_quality", False)):
        return "Low_Quality"
    return "Valid"


# ---------- Optional LLM fallback ----------
PROMPT_TEMPLATE = """You are labeling user reviews for policy compliance.
Pick exactly one label from this list:
- Advertisement
- Irrelevant
- Rant_No_Visit
- Low_Quality
- Valid

Review: "{text}"
Answer:"""

def make_llm_classifier(llm_model: str):
    if pipeline is None:
        raise RuntimeError("transformers is not installed. `pip install transformers` to enable LLM fallback.")
    return pipeline("text-generation", model=llm_model)

def prompt_classify(text: str, generator) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return "Low_Quality"
    prompt = PROMPT_TEMPLATE.format(text=text.strip())
    out = generator(prompt, max_new_tokens=12)[0]["generated_text"]
    for label in PRIORITY:  # first match wins
        if label in out:
            return label
    return "Valid"


# ---------- Public API ----------
def build_features_new(
    parquet_path: str,
    text_col: str = "text_clean",
    use_llm: bool = False,
    llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    tokenizer_model: Optional[str] = "distilbert-base-uncased",
    max_len: int = 128,
) -> Tuple[Dict[str, "torch.Tensor"], List[int], pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """
    Load the NEW dataset parquet and produce:
      - encodings: transformer tokenized inputs (dict of tensors)
      - labels: list[int] label ids
      - df: original dataframe with added columns 'label' and 'label_id'
      - label2id / id2label mappings

    If `use_llm=True`, rows not captured by policy flags are sent to a tiny LLM for a best-effort label.
    """
    df = pd.read_parquet(parquet_path)

    # 1) Heuristic label from new policy flags
    df["label"] = df.apply(assign_label_from_policies, axis=1)

    # 2) Optional LLM fallback ONLY for rows that ended up "Valid"
    generator = None
    if use_llm:
        generator = make_llm_classifier(llm_model)
        mask = df["label"].eq("Valid")
        if mask.any():
            df.loc[mask, "label"] = df.loc[mask, text_col].apply(lambda t: prompt_classify(t, generator))

    # 3) Map labels -> ids (stable ordering by PRIORITY)
    cat_type = pd.CategoricalDtype(categories=PRIORITY, ordered=True)
    df["label"] = df["label"].astype(cat_type)
    df["label_id"] = df["label"].cat.codes  # -1 means unseen; shouldn’t happen with given categories
    # Replace any -1 with "Valid" just in case
    df.loc[df["label_id"] == -1, "label"] = "Valid"
    df["label_id"] = df["label"].cat.codes

    label2id = {lab: i for i, lab in enumerate(df["label"].cat.categories)}
    id2label = {i: lab for lab, i in label2id.items()}

    # 4) Tokenize text (optional—skip if tokenizer_model=None)
    if tokenizer_model is None:
        encodings = {}
    else:
        if AutoTokenizer is None:
            raise RuntimeError("transformers is not installed. `pip install transformers` to create encodings.")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        encodings = tokenizer(
            df[text_col].fillna("").astype(str).tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )

    labels = df["label_id"].tolist()
    return encodings, labels, df, label2id, id2label


# ---------- CLI smoke test ----------
if __name__ == "__main__":
    parquet_path = "data/processed/reviews_wyoming.parquet"
    enc, y, df, label2id, id2label = build_features_new(
        parquet_path=parquet_path,
        use_llm=False,                      # set True only if you want LLM fallback
        tokenizer_model="distilbert-base-uncased",
        max_len=128
    )
    print("Label counts:\n", df["label"].value_counts(dropna=False))
    print("Label map:", label2id)
    print(df[[ "text_clean", "label" ]].head())
