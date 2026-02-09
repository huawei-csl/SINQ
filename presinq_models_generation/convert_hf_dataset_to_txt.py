# export_hf_to_txt.py
from datasets import load_dataset
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="e.g. openwebtext, ptb_text_only, allenai/c4")
    ap.add_argument("--config", default=None, help="optional config name")
    ap.add_argument("--split", default="validation", help="validation/test/train")
    ap.add_argument("--text_col", default=None, help="name of text column (if None, try to guess)")
    ap.add_argument("--out", required=True, help="output txt path")
    ap.add_argument("--max_rows", type=int, default=None, help="optional limit")
    args = ap.parse_args()

    ds = load_dataset(args.dataset, args.config, split=args.split)

    # Guess a text column if not provided
    text_col = args.text_col
    if text_col is None:
        for cand in ["text", "content", "document", "sentence", "article", "raw"]:
            if cand in ds.column_names:
                text_col = cand
                break
    if text_col is None:
        raise ValueError(f"Couldn't guess text column. Available columns: {ds.column_names}")

    n = len(ds) if args.max_rows is None else min(len(ds), args.max_rows)

    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(n):
            t = ds[i][text_col]
            if not isinstance(t, str):
                continue
            t = t.strip()
            if not t:
                continue
            # Separate docs with blank line to reduce accidental word-joining
            f.write(t.replace("\r\n", "\n").replace("\r", "\n"))
            f.write("\n\n")

    print(f"Wrote {n} rows from {args.dataset} ({args.split}) col={text_col} -> {args.out}")

if __name__ == "__main__":
    main()