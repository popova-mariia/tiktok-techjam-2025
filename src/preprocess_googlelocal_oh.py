# src/preprocess_googlelocal_oh.py
import argparse, re, sys, json, gzip, warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any
import pandas as pd
import numpy as np

try:
    from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
    HAVE_BS4 = True
except Exception:
    HAVE_BS4 = False

# ----------------------- Regex helpers -----------------------
URL_RE    = re.compile(r"\b((?:https?://|www\.)\S+)", re.I)
EMAIL_RE  = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE  = re.compile(r"(?:(?:\+?\d{1,3}[\s\-.]?)?(?:\(?\d{2,4}\)?[\s\-.]?)?\d{3}[\s\-.]?\d{4})")
WS_RE     = re.compile(r"\s+")
EXCL_RE   = re.compile(r"!")
Q_RE      = re.compile(r"\?")
ALLCAP_TOKEN_RE = re.compile(r"\b[A-ZÄÖÜİĞŞÇ]{2,}\b")
DIGIT_RE  = re.compile(r"\d")
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")

# Policies / categories
PROMO_RE = re.compile(
    r"\b(discount|promo|promotion|coupon|voucher|code|deal|% ?off|sale|offer|dm|whatsapp|we\s?chat|telegram)\b"
    r"| \b(buy\s?1\s?get\s?1|bogo)\b"
    r"| \b(use\s+code\s+[A-Z0-9]{3,})\b"
    r"| \b(call|contact)\s*\d{3,}\b"
    r"| https?:// | www\.",
    re.I | re.X
)

IRRELEVANT_RE = re.compile(
    r"\b(election|president|government|policy|war|politic(?:s|al)|campaign)\b"
    r"| \b(iphone|android|laptop|macbook|pc|gpu|camera)\b"
    r"| \b(stock|crypto|bitcoin|forex)\b",
    re.I | re.X
)

NO_VISIT_RE = re.compile(
    r"\b(never been|haven'?t (been|visited)|didn'?t (go|visit)|have not (been|visited)|"
    r"i didn'?t (visit|go)|we didn'?t (visit|go))\b",
    re.I
)

# Lexical buckets
POS_RE    = re.compile(r"\b(good|great|delicious|amazing|fantastic|love|wonderful|best|tasty|friendly)\b", re.I)
NEG_RE    = re.compile(r"\b(bad|terrible|horrible|worst|awful|hate|disgusting|poor|rude|slow)\b", re.I)
SERVICE_RE= re.compile(r"\b(service|staff|server|waiter|waitress|manager|rude|friendly)\b", re.I)
FOOD_RE   = re.compile(r"\b(food|dish|meal|taste|flavor|pizza|burger|sushi|noodle|coffee|drink)\b", re.I)
PRICE_RE  = re.compile(r"\b(price|expensive|cheap|value|deal|overpriced|affordable)\b", re.I)

# ------------- io -------------
def iter_json_lines(path: Path):
    if path.suffix == ".gz":
        f = gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    else:
        f = path.open("r", encoding="utf-8", errors="ignore")
    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def find_input_files(inp: Path) -> List[Path]:
    if inp.is_file():
        return [inp]
    files: List[Path] = []
    for p in inp.rglob("*"):
        if p.suffix in {".json", ".gz"}:
            files.append(p)
    files.sort()
    return files

# ------------- text utils -------------
def strip_html(text: str) -> Tuple[str, bool]:
    if not isinstance(text, str):
        return "", False
    if HAVE_BS4:
        had_html = ("<" in text and ">" in text) or "&" in text
        clean = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
        return clean, had_html
    had_html = ("<" in text and ">" in text)
    return re.sub(r"<[^>]+>", " ", text), had_html

def normalize(text: str) -> str:
    return WS_RE.sub(" ", (text or "").lower()).strip()

def parse_time(ts: Any):
    if ts is None or ts == "":
        return pd.NaT, -1, -1, -1
    t = pd.to_datetime(ts, errors="coerce", utc=True)
    if pd.isna(t):
        return pd.NaT, -1, -1, -1
    return t, int(t.year), int(t.month), int(t.dayofweek)

# ------------- main -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL(.json/.json.gz) file or directory.")
    ap.add_argument("--output", required=True, help="Output Parquet path.")
    ap.add_argument("--state_filter", default="OH", help="Filter by state (if present). Empty to disable.")
    ap.add_argument("--keep_raw", action="store_true", help="Keep text_raw column.")
    ap.add_argument("--drop_dupes", action="store_true", help="Drop exact dupes per (user_id,business_id,text_clean).")
    ap.add_argument("--sample_csv_rows", type=int, default=2000, help="Rows for preview CSV.")
    ap.add_argument("--debug_full", action="store_true", help="Keep all intermediate columns for debugging.")
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    files = find_input_files(inp)
    if not files:
        print(f"No input files found under {inp}", file=sys.stderr); sys.exit(1)

    rows = []
    sf = (args.state_filter or "").upper()

    for fp in files:
        for r in iter_json_lines(fp):
            user_id     = r.get("user_id") or r.get("user") or ""
            business_id = r.get("business_id") or r.get("gmap_id") or r.get("place_id") or ""
            rating      = r.get("rating", r.get("stars"))
            time_raw    = r.get("time") or r.get("date") or r.get("timestamp")
            text        = r.get("text") or r.get("content") or ""

            biz_name    = r.get("name") or r.get("business_name") or ""
            categories  = r.get("categories") or r.get("category") or ""
            city        = r.get("city") or ""
            state       = r.get("state") or r.get("province") or ""

            if sf and state and str(state).upper() != sf:
                continue

            stripped, had_html = strip_html(text)
            text_clean = normalize(stripped)

            has_url    = bool(URL_RE.search(text))
            num_urls   = len(URL_RE.findall(text))  # internal
            has_email  = bool(EMAIL_RE.search(text))
            has_phone  = bool(PHONE_RE.search(text))

            tokens         = text_clean.split() if text_clean else []
            length_tokens  = len(tokens)
            # raw counts (internal)
            num_exclaim    = len(EXCL_RE.findall(text))
            num_question   = len(Q_RE.findall(text))
            num_caps_tokens= len(ALLCAP_TOKEN_RE.findall(text))

            # ratios we keep
            avg_token_len      = (len(text_clean) / max(length_tokens, 1))
            ratio_exclaim      = (num_exclaim / max(length_tokens, 1))
            ratio_question     = (num_question / max(length_tokens, 1))
            caps_ratio         = (num_caps_tokens / max(length_tokens, 1))
            unique_token_ratio = (len(set(tokens)) / max(length_tokens, 1)) if length_tokens else 0.0

            # lexical buckets (keep only service/food/price)
            num_service = len(SERVICE_RE.findall(text_clean))
            num_food    = len(FOOD_RE.findall(text_clean))
            num_price   = len(PRICE_RE.findall(text_clean))

            # heuristic policy flags
            contains_promo      = bool(PROMO_RE.search(text))
            contains_irrelevant = bool(IRRELEVANT_RE.search(text))
            looks_rant_novisit  = bool(NO_VISIT_RE.search(text))
            is_empty_text       = (length_tokens == 0)
            is_short            = (length_tokens <= 10)

            ts, year, month, dow = parse_time(time_raw)

            row = {
                "user_id": str(user_id),
                "business_id": str(business_id),
                "business_name": str(biz_name),
                "categories_raw": str(categories),
                "city": str(city),
                "state": str(state),

                "rating": pd.to_numeric(rating, errors="coerce"),
                "time": ts, "year": year, "month": month, "day_of_week": dow,

                "text_clean": text_clean,
                "had_html": had_html,

                "has_url": has_url,
                "has_email": has_email,
                "has_phone": has_phone,

                "length_tokens": np.int32(length_tokens),
                "avg_token_len": float(avg_token_len),
                "ratio_exclaim": float(ratio_exclaim),
                "ratio_question": float(ratio_question),
                "caps_ratio": float(caps_ratio),
                "unique_token_ratio": float(unique_token_ratio),

                "num_service_words": np.int16(num_service),
                "num_food_words": np.int16(num_food),
                "num_price_words": np.int16(num_price),

                "contains_promo": contains_promo,
                "contains_irrelevant": contains_irrelevant,
                "looks_rant_novisit": looks_rant_novisit,
                "is_empty_text": is_empty_text,
                "is_short": is_short,
            }

            if args.keep_raw:
                row["text_raw"] = text

            # internal helpers (we'll drop unless --debug_full)
            row["_num_urls"]        = np.int16(num_urls)
            row["_num_exclaim"]     = np.int16(num_exclaim)
            row["_num_question"]    = np.int16(num_question)
            row["_num_caps_tokens"] = np.int16(num_caps_tokens)

            rows.append(row)

    if not rows:
        print("No rows after filtering.", file=sys.stderr); sys.exit(1)

    df = pd.DataFrame(rows)

    # aggregates
    for key, prefix in [("user_id","user"), ("business_id","biz")]:
        df = df.join(df.groupby(key, dropna=False).size().rename(f"{prefix}_review_count"), on=key)
        df = df.join(df.groupby(key, dropna=False)["rating"].mean().rename(f"{prefix}_avg_rating"), on=key)

    # dedupe (internal key)
    df["_review_fp"] = df["user_id"].astype(str) + "|" + df["business_id"].astype(str) + "|" + df["text_clean"].astype(str)
    if args.drop_dupes:
        df = df.drop_duplicates(subset=["_review_fp"], keep="first").copy()

    # policy bundles
    df["policy_ads"]         = (df["contains_promo"] | df["has_url"] | df["has_email"] | df["has_phone"]).astype("boolean")
    df["policy_irrelevant"]  = df["contains_irrelevant"].astype("boolean")
    df["policy_novisit"]     = df["looks_rant_novisit"].astype("boolean")
    df["policy_low_quality"] = (df["is_empty_text"] | df["is_short"]).astype("boolean")
    df["policy_violation_any"] = (df["policy_ads"] | df["policy_irrelevant"] | df["policy_novisit"] | df["policy_low_quality"]).astype("boolean")

    # stable id
    df["review_id"] = pd.util.hash_pandas_object(df["_review_fp"], index=False).astype("int64").astype("string")

    # decide final columns
    KEEP_COLS = [
        "review_id",
        "user_id","business_id","business_name","categories_raw","city","state",
        "rating","time","year","month","day_of_week",
        "text_clean","had_html",
        "has_url","has_email","has_phone",
        "length_tokens","avg_token_len","ratio_exclaim","ratio_question","caps_ratio","unique_token_ratio",
        "num_service_words","num_food_words","num_price_words",
        "policy_ads","policy_irrelevant","policy_novisit","policy_low_quality","policy_violation_any",
        "user_review_count","user_avg_rating","biz_review_count","biz_avg_rating"
    ]
    if args.keep_raw:
        KEEP_COLS.insert(KEEP_COLS.index("text_clean")+1, "text_raw")

    if not args.debug_full:
        df = df[KEEP_COLS]
    else:
        pass

    # save
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outp, index=False)
    df.head(min(args.sample_csv_rows, len(df))).to_csv(outp.with_suffix(".sample.csv"), index=False)

        # Console summary
    print(f"Saved: {outp}  rows={len(df)}  users={df['user_id'].nunique()}  businesses={df['business_id'].nunique()}")
    print("Policy counts:", {
        "ads": int(df["policy_ads"].sum()),
        "irrelevant": int(df["policy_irrelevant"].sum()),
        "no_visit": int(df["policy_novisit"].sum()),
        "low_quality": int(df["policy_low_quality"].sum()),
        "any": int(df["policy_violation_any"].sum()),
    })

    PREVIEW_N = 10           # how many rows to show
    TEXT_W    = 120          # max chars for text preview

    # pick only the essentials for console
    cols_to_show = [
        "user_id", "business_id", "city", "state",
        "rating", "time",
        "policy_ads", "policy_irrelevant", "policy_novisit", "policy_low_quality",
        "length_tokens", "avg_token_len",
        "num_service_words", "num_food_words", "num_price_words",
        "text_clean",
    ]

    preview = df.loc[:, [c for c in cols_to_show if c in df.columns]].head(PREVIEW_N).copy()
    # truncate long text
    preview["text_clean"] = preview["text_clean"].astype(str).apply(
        lambda s: (s[:TEXT_W] + "…") if len(s) > TEXT_W else s
    )

    # print a compact table
    with pd.option_context("display.max_rows", PREVIEW_N,
                           "display.max_columns", None,
                           "display.width", 160,
                           "display.colheader_justify", "left"):
        print("\nPreview:")
        print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
