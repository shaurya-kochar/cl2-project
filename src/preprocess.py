#!/usr/bin/env python3
"""
Preprocessing script for IMDb dataset.
Produces: data/processed.csv and data/meta.json
Run: python src/preprocess.py --input data/IMDB_Dataset.csv --output data/processed.csv
"""

import argparse
import json
import os
import re
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag


URL_RE = re.compile(r'https?://\S+|www\.\S+')
HTML_TAG_RE = re.compile(r'<.*?>')
NON_ALPHANUMERIC_RE = re.compile(r'[^a-zA-Z0-9\s\.\,\!\?\'"]') 
MULTI_SPACE_RE = re.compile(r'\s+')

NEGATIONS = {"no", "not", "never", "n't", "nothing", "nowhere", "neither", "nor"}


def clean_text(text: str, keep_punct: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    t = text
    t = t.replace("\r", " ").replace("\n", " ")
    t = HTML_TAG_RE.sub(" ", t)
    t = URL_RE.sub(" ", t)
    if keep_punct:
        t = NON_ALPHANUMERIC_RE.sub(" ", t)
    else:
        t = re.sub(r'[^a-zA-Z0-9\s]', ' ', t)
    t = MULTI_SPACE_RE.sub(" ", t)
    t = t.strip().lower()
    return t


def tokenize_text(text: str) -> list:
    try:
        toks = word_tokenize(text)
    except Exception:
        toks = text.split()
    toks = [t for t in toks if t.strip() != ""]
    return toks


def pos_tag_tokens(tokens: list) -> list:
    if not tokens:
        return []
    try:
        tagged = pos_tag(tokens)
        return [p for (w, p) in tagged]
    except Exception:
        return ["NN"] * len(tokens)


def punctuation_count(text: str) -> int:
    return sum(1 for ch in text if ch in {'.', ',', '!', '?', ';', ':'})


def negation_count(tokens: list) -> int:
    return sum(1 for t in tokens if t.lower() in NEGATIONS)


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "review",
                         label_col: str = "sentiment",
                         keep_punct: bool = True,
                         pos_sample_limit: int = None):
    """
    pos_sample_limit: if int, will POS tag only the first N rows to save time; use None to tag all
    """
    records = []
    n = len(df)
    it = tqdm(df.itertuples(index=False), total=n, desc="Preprocessing")
    def map_label(l):
        if isinstance(l, str):
            l2 = l.strip().lower()
            if l2 in {"positive", "pos", "1", "p"}:
                return 1
            if l2 in {"negative", "neg", "0", "n"}:
                return 0
        try:
            return int(l)
        except Exception:
            return l

    for i, row in enumerate(it):
        text_raw = getattr(row, text_col)
        label_raw = getattr(row, label_col) if hasattr(row, label_col) else None
        label = map_label(label_raw)

        text_clean = clean_text(text_raw, keep_punct=keep_punct)
        tokens = tokenize_text(text_clean)
        if (pos_sample_limit is None) or (i < pos_sample_limit):
            pos_tags = pos_tag_tokens(tokens)
        else:
            pos_tags = []

        tlen = len(tokens)
        avg_wlen = np.mean([len(t) for t in tokens]) if tokens else 0.0
        punc_cnt = punctuation_count(text_raw)
        neg_cnt = negation_count(tokens)

        rec = {
            "id": i,
            "text_raw": text_raw,
            "text_clean": text_clean,
            "tokens": tokens,
            "pos_tags": pos_tags,
            "text_length": tlen,
            "avg_word_len": float(avg_wlen),
            "punct_count": int(punc_cnt),
            "negation_count": int(neg_cnt),
            "label": label
        }
        records.append(rec)

    proc_df = pd.DataFrame.from_records(records)
    return proc_df


def main(args):
    inp = Path(args.input)
    out = Path(args.output)
    meta_out = Path(args.meta_output) if args.meta_output else out.parent / "meta.json"

    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")

    print("Loading:", inp)
    df = pd.read_csv(inp)
    expected_cols = set(df.columns)
    print("Columns:", expected_cols)

    proc_df = preprocess_dataframe(df,
                                   text_col=args.text_col,
                                   label_col=args.label_col,
                                   keep_punct=not args.remove_punct,
                                   pos_sample_limit=(None if not args.pos_limit else int(args.pos_limit)))

 
    proc_df_copy = proc_df.copy()
    proc_df_copy["tokens"] = proc_df_copy["tokens"].apply(json.dumps)
    proc_df_copy["pos_tags"] = proc_df_copy["pos_tags"].apply(json.dumps)
    proc_df_copy.to_csv(out, index=False, encoding="utf-8")
    print("Saved processed CSV:", out)

    meta = {
        "n_raw": int(len(df)),
        "n_processed": int(len(proc_df)),
        "label_counts": dict(pd.Series(proc_df['label']).value_counts().to_dict()),
        "text_length": {
            "mean": float(proc_df["text_length"].mean()),
            "median": float(proc_df["text_length"].median()),
            "max": int(proc_df["text_length"].max()),
            "min": int(proc_df["text_length"].min())
        }
    }
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Saved meta:", meta_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/IMDB_Dataset.csv", help="input raw CSV path")
    parser.add_argument("--output", default="data/processed.csv", help="output processed CSV")
    parser.add_argument("--meta-output", default=None, help="output meta JSON path")
    parser.add_argument("--text-col", default="review", help="text column name")
    parser.add_argument("--label-col", default="sentiment", help="label column name")
    parser.add_argument("--remove-punct", action="store_true", help="remove punctuation (default keep)")
    parser.add_argument("--pos-limit", default=None,
                        help="if set, only POS-tag this many rows (useful for speed), pass int")
    args = parser.parse_args()
    main(args)
