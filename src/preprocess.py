# scripts/preprocess.py
import argparse, re, sys, json
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# ---------- regex helpers ----------
URL_RE = re.compile(r"""(?i)\b((?:https?://|www\.)\S+)""")
WS_RE  = re.compile(r"\s+")
EXCL_RE = re.compile(r"!")
Q_RE    = re.compile(r"\?")
# crude all-caps "token" detector (≥2 letters, all caps)
ALLCAP_TOKEN_RE = re.compile(r"\b[A-ZÄÖÜİĞŞÇ]{2,}\b")

def strip_html(text: str) -> str:
    if not isinstance(text, str): return ""
    return BeautifulSoup(text, "html.parser").get_text(" ", strip=True)

def remove_urls(text: str) -> tuple[str, bool]:
    if not isinstance(text, str): return "", False
    has = bool(URL_RE.search(text))
    return URL_RE.sub("", text), has

def normalize(text: str) -> str:
    text = text.lower()
    text = WS_RE.sub(" ", text).strip()
    return text

def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Input CSV path")
    ap.add_argument("--output", type=str, required=True, help="Output parquet path")
    ap.add_argument("--labelmap_out", type=str, default="", help="Optional JSON path to save category_id↔name map")
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    df = load_csv(inp)

    required = ["business_name", "author_name", "text", "photo", "rating", "rating_category"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("Missing required columns:", missing, file=sys.stderr)
        print("Available columns:", list(df.columns), file=sys.stderr)
        sys.exit(1)

    # base frame
    out = pd.DataFrame()
    out["business_name"] = df["business_name"].astype(str)
    out["author_name"]   = df["author_name"].astype(str)
    out["photo_path"]    = df["photo"].astype(str)
    out["has_photo"]     = out["photo_path"].str.len().fillna(0).astype(int).gt(0)

    # rating: coerce to numeric and clip to [1,5]
    rating_num = pd.to_numeric(df["rating"], errors="coerce")
    out["rating"] = rating_num.clip(lower=1, upper=5)

    # class label
    out["category_raw"] = df["rating_category"].astype(str)
    cat = pd.Categorical(out["category_raw"])
    out["category_id"] = cat.codes.astype("int16")  # -1 if NaN, but we cast after fillna below

    # text cleaning
    raw = df["text"].fillna("")
    stripped = raw.apply(strip_html)
    no_url_and_flag = stripped.apply(remove_urls)
    out["text_raw"]   = raw
    out["text_clean"] = [normalize(x[0]) for x in no_url_and_flag]
    out["has_url"]    = [x[1] for x in no_url_and_flag]

    # lengths & basic signals
    out["length_chars"]  = out["text_clean"].str.len().fillna(0).astype(int)
    out["length_tokens"] = out["text_clean"].str.split().apply(len).astype(int)
    out["num_exclaim"]   = out["text_raw"].fillna("").apply(lambda s: len(EXCL_RE.findall(s)))
    out["num_question"]  = out["text_raw"].fillna("").apply(lambda s: len(Q_RE.findall(s)))
    out["num_caps_tokens"]= out["text_raw"].fillna("").apply(lambda s: len(ALLCAP_TOKEN_RE.findall(s)))
    out["is_short"]      = out["length_tokens"] <= 10

    # ids, time
    out["timestamp"] = None
    
    out["review_id"] = pd.util.hash_pandas_object(
        out["business_name"].str.cat(out["author_name"], sep="|").str.cat(out["text_clean"], sep="|"),
        index=False
    ).astype("int64").astype("string")

    # drop rows with empty cleaned text
    out = out[out["text_clean"].str.len() > 0].copy()

    # drop exact dupes by (business, author, text)
    out = out.drop_duplicates(subset=["business_name", "author_name", "text_clean"], keep="first")

    # tidy dtypes
    out["category_id"] = out["category_id"].replace({-1: pd.NA}).astype("Int16")

    # save
    out.to_parquet(outp, index=False)

    # label map
    if args.labelmap_out:
        lm_path = Path(args.labelmap_out)
        lm_path.parent.mkdir(parents=True, exist_ok=True)
        # keep ordering consistent with categorical categories_
        label_map = {int(i): name for i, name in enumerate(list(cat.categories))}
        with open(lm_path, "w", encoding="utf-8") as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)

    # csv sample
    sample_csv = outp.with_suffix(".sample.csv")
    out.head(2000).to_csv(sample_csv, index=False)

    # summary
    print(f"Saved: {outp}  rows={len(out)}  uniques_business={out['business_name'].nunique()}")

    cols_to_show = ["business_name", "author_name", "rating", "category_raw", "text_clean"]
    print("Preview:")
    print(out[cols_to_show].sample(min(5, len(out))).to_string(index=False))


if __name__ == "__main__":
    main()
