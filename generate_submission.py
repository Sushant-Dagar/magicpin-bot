#!/usr/bin/env python3
"""
Generate submission.jsonl from the 30 canonical (merchant, trigger[, customer]) test pairs.

Usage:
    # 1. Expand the seed dataset to the full 50/200/100 set + test_pairs.json
    #    (only needs to be run once; deterministic, same output for everyone):
    python dataset/generate_dataset.py --out ./expanded

    # 2. Set your LLM provider + key so compose() uses the real LLM path, not the
    #    rule-based fallback (the fallback is schema-safe but scores much lower on
    #    specificity/engagement — only use it as a last resort):
    export LLM_PROVIDER=groq        # or openai / anthropic
    export GROQ_API_KEY=...         # or OPENAI_API_KEY / ANTHROPIC_API_KEY
    export LLM_MODEL=llama-3.3-70b-versatile

    # 3. Generate:
    python generate_submission.py --expanded ./expanded --out submission.jsonl
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from composer import compose  # noqa: E402


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expanded", default="./expanded", help="Path to expanded dataset dir")
    ap.add_argument("--out", default="submission.jsonl")
    args = ap.parse_args()

    root = Path(args.expanded)
    test_pairs = load_json(root / "test_pairs.json")["pairs"]

    categories = {}
    for f in (root / "categories").glob("*.json"):
        d = load_json(f)
        categories[d["slug"]] = d

    merchants = {}
    for f in (root / "merchants").glob("*.json"):
        d = load_json(f)
        merchants[d["merchant_id"]] = d

    customers = {}
    for f in (root / "customers").glob("*.json"):
        d = load_json(f)
        customers[d["customer_id"]] = d

    triggers = {}
    for f in (root / "triggers").glob("*.json"):
        d = load_json(f)
        triggers[d["id"]] = d

    provider = os.getenv("LLM_PROVIDER", "openai")
    has_key = any(os.getenv(k) for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY"))
    if not has_key:
        print("WARNING: no LLM API key set in environment. Every line below will use the "
              "rule-based FALLBACK composer, which is schema-safe but will score noticeably "
              "lower on Specificity/Category Fit/Engagement than the real LLM path. Set "
              "LLM_PROVIDER + the matching *_API_KEY before generating your real submission.",
              file=sys.stderr)

    lines = []
    fallback_count = 0
    for tp in test_pairs:
        test_id = tp["test_id"]
        trigger = triggers.get(tp["trigger_id"])
        merchant = merchants.get(tp["merchant_id"])
        customer = customers.get(tp["customer_id"]) if tp.get("customer_id") else None
        if not trigger or not merchant:
            print(f"SKIP {test_id}: missing trigger or merchant in dataset", file=sys.stderr)
            continue
        category = categories.get(merchant.get("category_slug", ""))
        if not category:
            print(f"SKIP {test_id}: missing category for merchant {merchant.get('merchant_id')}", file=sys.stderr)
            continue

        result = compose(category, merchant, trigger, customer)
        if "[LLM error" in result.get("rationale", ""):
            fallback_count += 1

        lines.append({
            "test_id": test_id,
            "body": result.get("body", ""),
            "cta": result.get("cta", "none"),
            "send_as": result.get("send_as", "vera"),
            "suppression_key": result.get("suppression_key", trigger.get("suppression_key", "")),
            "rationale": result.get("rationale", ""),
        })
        print(f"{test_id}: {result.get('body', '')[:90]}")

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(lines)} lines to {out_path}")
    if fallback_count:
        print(f"WARNING: {fallback_count}/{len(lines)} lines used the fallback composer "
              f"(LLM call failed or no key set). Fix your LLM_PROVIDER/API key and re-run "
              f"before submitting.", file=sys.stderr)


if __name__ == "__main__":
    main()
