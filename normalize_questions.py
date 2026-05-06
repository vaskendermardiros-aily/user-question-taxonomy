"""Prepare a normalized question dataset for EDA.

This script reads `data/unique_questions.csv`, keeps only rows where
`chat_origin == "free_question"`, builds an `effective_question` by using
`override_question` when present and `question` otherwise, normalizes
whitespace and Unicode formatting, and removes rows that are empty or too
short after normalization.

The cleaned dataset is written to `data/unique_questions_normalized.csv`.
"""

import csv
import re
import unicodedata
from pathlib import Path

MIN_QUESTION_LENGTH = 6


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_effective_question(row: dict[str, str]) -> str:
    override_question = normalize_text(row.get("override_question", ""))
    question = normalize_text(row.get("question", ""))
    return override_question or question


def process_file(input_path: Path, output_path: Path) -> None:
    kept_rows = []
    total_rows = 0
    free_question_rows = 0
    removed_empty = 0
    removed_too_short = 0

    with input_path.open("r", newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)

        for row in reader:
            total_rows += 1

            if row.get("chat_origin") != "free_question":
                continue

            free_question_rows += 1
            effective_question = build_effective_question(row)

            if not effective_question:
                removed_empty += 1
                continue

            if len(effective_question) < MIN_QUESTION_LENGTH:
                removed_too_short += 1
                continue

            cleaned_row = dict(row)
            cleaned_row["effective_question"] = effective_question
            kept_rows.append(cleaned_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "question",
        "override_question",
        "effective_question",
        "created_at",
        "chat_origin",
        "sequence_number",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    print(f"Input rows: {total_rows}")
    print(f"free_question rows: {free_question_rows}")
    print(f"Removed empty effective questions: {removed_empty}")
    print(f"Removed questions shorter than {MIN_QUESTION_LENGTH} chars: {removed_too_short}")
    print(f"Saved cleaned rows: {len(kept_rows)}")
    print(f"Output: {output_path}")

if __name__ == "__main__":
    input_path = Path("data/unique_questions.csv")
    output_path = Path("data/unique_questions_normalized.csv")
    process_file(input_path, output_path)
