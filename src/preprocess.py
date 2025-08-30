# src/preprocess.py
import argparse, re, sys, json, warnings
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# ---------- regex helpers ----------
URL_RE    = re.compile(r"(?i)\b((?:https?://|www\.)\S+)")
WS_RE     = re.compile(r"\s+")
EXCL_RE   = re.compile(r"!")
Q_RE      = re.compile(r"\?")
ALLCAP_TOKEN_RE = re.compile(r"\b[A-ZÄÖÜİĞŞÇ]{2,}\b")

EMAIL_RE  = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE_RE  = re.compile(r"(?:(?:\+?\d{1,3}[\s\-\.]?)?(?:\(?\d{2,4}\)?[\s\-\.]?)?\d{3}[\s\-\.]?\d{4})")
COUPON_RE = re.compile(r"(?i)\b(discount|promo|coupon|code|deal|% ?off|sale|offer|dm|whatsapp)\b")
NO_VISIT_RE = re.compile(r"(?i)\b(never been|haven'?t visited|didn'?t go|have not been|i didn'?t visit|i have not visited)\b")
DEVICE_RE   = re.compile(r"(?i)\b(iphone|android|laptop|macbook|pc|gpu|camera)\b")
POLITICS_RE = re.compile(r"(?i)\b(election|president|government|policy|war|party)\b")

def strip_html(text: str) -> tuple[str, bool]:
    """Return (plain_text, had_html_flag)."""
    if not isinstance(text, str): return "", False
    had_html = ("<" in text and ">" in text) or "&" in text
    clean = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    return clean, had_html

def remove_urls(text: str) -> tuple[str, bool, int]:
    if not isinstance(text, str): return "", False, 0
    urls = URL_RE.findall(text)
    no_url = URL_RE.sub("", text)
    return no_url, bool(urls), len(urls)

def normalize(text: str) -> str:
    text = (text or "").lower()
    return WS_RE.sub(" ", text).strip()

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
    # new flags: keep vs drop behaviours
    ap.add_argument("--drop_empty_clean", action="store_true", help="If set, drop rows where text_clean is empty")
    ap.add_argument("--drop_dup_exact", action="store_true", help="If set, drop exact duplicates (biz, author, text_clean)")
    ap.add_argument("--verbose_preview", action="store_true", help="If set, preview will include extra debug columns")

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
    out["category_id"] = pd.Series(cat.codes, dtype="int16")  # -1 if NaN, but we cast after fillna below

    # text cleaning
    raw = df["text"]
    out["text_raw"] = raw

    # strip HTML (but keep content intact)
    stripped_pairs = raw.apply(strip_html)  # -> (stripped, had_html)
    stripped_text  = [p[0] for p in stripped_pairs]
    had_html_flags = [p[1] for p in stripped_pairs]

    # URL detection only (don’t remove from text)
    has_url_flag = [bool(URL_RE.search(t)) if isinstance(t,str) else False for t in stripped_text]
    num_urls     = [len(URL_RE.findall(t)) if isinstance(t,str) else 0 for t in stripped_text]

    out["text_stripped"] = stripped_text
    out["had_html"]      = pd.Series(had_html_flags, dtype="boolean")
    out["text_clean"]    = pd.Series(stripped_text).apply(normalize)   # lowercased + ws-normalized, but URLs preserved
    out["has_url"]       = pd.Series(has_url_flag, dtype="boolean")
    out["num_urls"]      = pd.Series(num_urls, dtype="Int16")

    # additional raw signals
    tr = out["text_raw"].fillna("")
    out["has_email"]      = tr.str.contains(EMAIL_RE).fillna(False).astype("boolean")
    out["has_phone"]      = tr.str.contains(PHONE_RE).fillna(False).astype("boolean")
    out["contains_coupon"] = tr.str.contains(COUPON_RE).fillna(False).astype("boolean")

    # lengths & counts (use clean or raw as appropriate)
    out["length_chars"]   = out["text_clean"].str.len().fillna(0).astype(int)
    out["length_tokens"]  = out["text_clean"].str.split().apply(len).astype(int)
    out["num_exclaim"]    = tr.apply(lambda s: len(EXCL_RE.findall(s))).astype(int)
    out["num_question"]   = tr.apply(lambda s: len(Q_RE.findall(s))).astype(int)
    out["num_caps_tokens"]= tr.apply(lambda s: len(ALLCAP_TOKEN_RE.findall(s))).astype(int)
    out["is_short"]       = (out["length_tokens"] <= 10)

     # --- extra derived numeric features ---
    out["avg_token_len"] = (
        out["length_chars"] / out["length_tokens"].clip(lower=1)
    ).astype(float)

    out["ratio_exclaim"] = (
        out["num_exclaim"] / out["length_tokens"].clip(lower=1)
    ).astype(float)
    out["ratio_question"] = (
        out["num_question"] / out["length_tokens"].clip(lower=1)
    ).astype(float)

    out["caps_ratio"] = (
        out["num_caps_tokens"] / out["length_tokens"].clip(lower=1)
    ).astype(float)

    out["unique_token_ratio"] = out["text_clean"].apply(
        lambda s: len(set(str(s).split())) if isinstance(s, str) else 0
    ) / out["length_tokens"].clip(lower=1)

    POS_RE = re.compile(r"\b(good|great|delicious|amazing|fantastic|love|wonderful|best)\b", re.I)
    NEG_RE = re.compile(r"\b(bad|terrible|horrible|worst|awful|hate|disgusting|poor)\b", re.I)

    out["num_pos_words"] = out["text_clean"].str.count(POS_RE).fillna(0).astype(int)
    out["num_neg_words"] = out["text_clean"].str.count(NEG_RE).fillna(0).astype(int)

    # initial flags (key change)
    out["is_empty_text_raw"]   = out["text_raw"].isna() | (out["text_raw"].astype(str).str.strip() == "")
    out["is_empty_text_clean"] = out["text_clean"].astype(str).str.len() == 0
    out["is_rating_only"]      = out["is_empty_text_clean"] & out["rating"].notna()

    # light heuristics (still not final labels)
    out["looks_promo_heur"]        = (out["has_url"] | out["has_email"] | out["has_phone"] | out["contains_coupon"]).astype("boolean")
    out["looks_rant_no_visit_heur"]= tr.str.contains(NO_VISIT_RE).fillna(False).astype("boolean")
    out["looks_irrelevant_heur"]   = (tr.str.contains(DEVICE_RE) | tr.str.contains(POLITICS_RE)).fillna(False).astype("boolean")
    out["looks_low_quality_heur"] = (out["is_short"] | out["is_rating_only"] | out["is_empty_text_clean"]).astype("boolean")

    # ids
    out["timestamp"] = pd.NaT  # placeholder, parse if you add a date column later
    out["review_id"] = pd.util.hash_pandas_object(
        out["business_name"].str.cat(out["author_name"], sep="|").str.cat(out["text_clean"], sep="|"),
        index=False
    ).astype("int64").astype("string")

    # duplicates: mark instead of blindly dropping
    key = ["business_name","author_name","text_clean"]
    grp = out.groupby(key, dropna=False)
    out["dupe_count"]  = grp["review_id"].transform("size").astype("Int16")
    out["is_exact_dupe"]= (out["dupe_count"] > 1)

    # optional dropping (controlled by flags)
    if args.drop_empty_clean:
        out = out[~out["is_empty_text_clean"]].copy()
    if args.drop_dup_exact:
        out = out.drop_duplicates(subset=key, keep="first").copy()

    # tidy dtypes
    out["category_id"] = out["category_id"].replace({-1: pd.NA}).astype("Int16")

    # save
    out.to_parquet(outp, index=False)

    # label map
    if args.labelmap_out:
        lm_path = Path(args.labelmap_out)
        lm_path.parent.mkdir(parents=True, exist_ok=True)
        label_map = {int(i): name for i, name in enumerate(list(cat.categories))}
        with open(lm_path, "w", encoding="utf-8") as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)

    # csv sample
    sample_csv = outp.with_suffix(".sample.csv")
    out.head(2000).to_csv(sample_csv, index=False)

    # summary
    # for now showing all the cols for debugging purposes 
    print(f"Saved: {outp} rows={len(out)} uniques_business={out['business_name'].nunique()}")
    cols_to_show = ["business_name","author_name","rating","category_raw",
                    "has_url","has_email","has_phone","contains_coupon",
                    "length_tokens","avg_token_len","caps_ratio","unique_token_ratio",
                    "num_pos_words","num_neg_words",
                    "is_empty_text_clean","is_rating_only","is_exact_dupe",
                    "looks_promo_heur","looks_rant_no_visit_heur","looks_irrelevant_heur","looks_low_quality_heur",
                    "text_clean"]
    print("Preview:")
    print(out[cols_to_show].sample(min(5, len(out))).to_string(index=False))


if __name__ == "__main__":
    main()
