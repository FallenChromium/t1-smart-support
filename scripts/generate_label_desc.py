"""
Quick helper to build a label description CSV via SciBox-hosted Qwen.

Usage:
    python scripts/generate_label_desc.py --data data.csv --output label_desc.csv

Requirements:
    - SCIBOX_API_KEY environment variable set.
    - Optional SCIBOX_BASE_URL / SCIBOX_QWEN_MODEL overrides.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from pathlib import Path
from textwrap import shorten

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


def _top_keywords(texts: pd.Series, limit: int = 5) -> list[str]:
    """Return most common lowercase tokens (length >= 4) across provided texts."""
    counter: Counter[str] = Counter()
    for text in texts.dropna().astype(str):
        tokens = re.findall(r"[A-Za-zА-Яа-я0-9\-']+", text.lower())
        for token in tokens:
            normalized = token.strip("-' ")
            if len(normalized) >= 4 and not normalized.isdigit():
                counter[normalized] += 1
    return [token for token, _ in counter.most_common(limit)]


def build_prompt(coarse: str, fine: str, rows: pd.DataFrame) -> str:
    """Construct a Russian prompt using representative ticket bodies."""
    question_series = rows.get("query_text", pd.Series(dtype=str))
    question_series = question_series.dropna().astype(str)
    most_common_questions = list(question_series.value_counts().head(5).index)
    if len(most_common_questions) < 5:
        # Fall back to preserving input order if value counts are too sparse.
        for text in question_series.head(5):
            if text not in most_common_questions:
                most_common_questions.append(text)
            if len(most_common_questions) >= 5:
                break

    examples = [
        f"- {shorten(text, width=160, placeholder='…')}"
        for text in most_common_questions[:5]
    ]
    examples_str = "\n".join(examples) if examples else "- (нет примеров в данных)"

    keywords = _top_keywords(question_series, limit=5)
    keywords_line = (
        "Key user inent: " + ", ".join(keywords) if keywords else ""
    )

    return (
        "You are a support analyst who formulates short and accurate names for ticket subcategories.\n"
        f"Top-level category: {coarse}\n"
        f"Subcategory from source data: {fine}\n"
        f"{keywords_line}\n"
        "Below are typical query texts (truncated to 160 characters). "
        "Concentrate on the recurring customer pain points, not on rare details.\n"
        "Task: come up with one English name (up to 8 words) that reflects "
        "the main plot of the majority of queries in this subcategory. Do not use quotation marks, "
        "numbers, or a final period.\n"
        "Queries:\n"
        f"{examples_str}\n"
        "Output only the name."
    )


def call_qwen(client: OpenAI, prompt: str) -> str:
    """Query Qwen for a short Russian label."""
    system = (
        "You come up with concise names for support ticket subcategories. Always answer in English in a single line, without quotation marks or a final period."
    )
    resp = client.chat.completions.create(
        model=os.getenv("SCIBOX_QWEN_MODEL", "Qwen2.5-72B-Instruct-AWQ"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=float(os.getenv("SCIBOX_CHAT_TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("SCIBOX_CHAT_MAX_TOKENS", "64")),
    )
    return (resp.choices[0].message.content or "").strip()


def generate_labels(df: pd.DataFrame, client: OpenAI) -> pd.DataFrame:
    """Generate a dataframe with category, subcategory, and generated_label."""
    records: list[dict[str, str]] = []
    grouped = df.groupby(["category", "subcategory"], sort=False)
    for (coarse, fine), rows in grouped:
        prompt = build_prompt(str(coarse), str(fine), rows)
        label = call_qwen(client, prompt)
        if not label:
            label = f"{coarse} {fine}".strip()
        cleaned_label = shorten(label, width=160, placeholder="…")
        records.append(
            {
                "category": str(coarse),
                "subcategory": str(fine),
                "generated_label": cleaned_label,
            }
        )
    return pd.DataFrame.from_records(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate category/subcategory labels via SciBox Qwen."
    )
    parser.add_argument(
        "--data", type=Path, default=Path("tickets.csv"), help="Input CSV with FAQ data."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("label_desc.csv"),
        help="Destination CSV file with columns category, subcategory, generated_label.",
    )
    parser.add_argument(
        "--separator", type=str, default=",", help="CSV separator (default ',')."
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    api_key = os.getenv("SCIBOX_API_KEY")
    if not api_key:
        raise RuntimeError("SCIBOX_API_KEY is not set. Export it before running.")

    args = parse_args()
    df = pd.read_csv(args.data, sep=args.separator)
    expected_cols = {"category", "subcategory"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {args.data}")

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("SCIBOX_BASE_URL", "https://llm.t1v.scibox.tech/v1"),
    )
    labels_df = generate_labels(df, client)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(args.output, index=False)
    print(f"Wrote {len(labels_df)} label descriptions to {args.output}")


if __name__ == "__main__":
    main()
