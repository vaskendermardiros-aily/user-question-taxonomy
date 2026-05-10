"""Prepare a normalized question dataset for EDA.

This script reads `data/unique_questions.csv`, keeps only rows where
`chat_origin == "free_question"`, builds an `effective_question` by using
`override_question` when present and `question` otherwise, normalizes
whitespace and Unicode formatting, and removes rows that are empty or too
short after normalization.

The cleaned dataset is written to `data/unique_questions_normalized.csv`.
"""

import json
import re
import unicodedata
from pathlib import Path
import pandas as pd
import logging

_logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

MIN_QUESTION_LENGTH = 6

# Paths
_THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = _THIS_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR = DATA_DIR / "raw"
NORMALIZED_DATA_DIR = DATA_DIR / "normalized"
NORMALIZED_DATA_DIR.mkdir(parents=True, exist_ok=True)
USERS_CSV_PATH = DATA_DIR / "users.csv"

_CHAT_ORIGINS = ["free_question", "auto_suggestion", "followup", "auto_followup"]


def normalize_text(text: str) -> str:
    if not text or str(text) == "nan":
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_effective_question(row: dict[str, str]) -> str:
    override_question = normalize_text(row.get("override_question", ""))
    question = normalize_text(row.get("question", ""))
    return override_question or question


def build_effective_answer(row: dict[str, str]) -> str:
    answer = row.get("answer", None)
    if answer is None:
        return ""
    summary = normalize_text(answer.get("summary", ""))
    key_points = normalize_text(answer.get("key_points", ""))
    return summary + " " + key_points


def process_file(input_path: Path, output_path: Path) -> None:
    # kept_rows = []
    # total_rows = 0
    # free_question_rows = 0
    # removed_empty = 0
    # removed_too_short = 0

    # Load the data into a pandas dataframe
    df = pd.read_json(input_path, lines=True)

    if len(df) == 0:
        _logger.warning(f"No rows in {input_path}")
        return

    # Filter the dataframe to only include rows where chat_origin is in _CHAT_ORIGINS
    df = df[df["chat_origin"].isin(_CHAT_ORIGINS)]

    # Build the effective question
    df["effective_question"] = df.apply(build_effective_question, axis=1)

    # Answer
    df["effective_answer"] = df.apply(build_effective_answer, axis=1)

    # # NOTE: want to keep the full conversation...
    # # Drop baddies
    # # Filter the dataframe to only include rows where effective_question is not empty
    # df = df[df["effective_question"].notna()]

    # # Filter the dataframe to only include rows where effective_question is longer than MIN_QUESTION_LENGTH
    # df = df[df["effective_question"].str.len() >= MIN_QUESTION_LENGTH]

    # Columns to keep
    columns_to_keep = [
        "interaction_id",
        "req_user_id",
        "sequence_number",
        "chat_origin",
        # "question",
        # "override_question",
        "effective_question",
        "effective_answer",
        "req_created_at",
        "req_id",
        "resp_created_at",
        "resp_id",
    ]

    # Save the dataframe to a JSONL file
    records = df[columns_to_keep].to_dict(orient="records")
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")

    # print(f"Input rows: {total_rows}")
    # print(f"free_question rows: {free_question_rows}")
    # print(f"Removed empty effective questions: {removed_empty}")
    # print(f"Removed questions shorter than {MIN_QUESTION_LENGTH} chars: {removed_too_short}")
    # print(f"Saved cleaned rows: {len(kept_rows)}")


if __name__ == "__main__":
    raw_data_files = RAW_DATA_DIR.glob("*.jsonl")

    for raw_data_file in raw_data_files:
        raw_data_file_name = raw_data_file.stem
        normalized_data_file = NORMALIZED_DATA_DIR / f"{raw_data_file_name}.jsonl"

        _logger.info(f"Processing {raw_data_file}...")
        process_file(raw_data_file, normalized_data_file)
        _logger.info(f"Processed {raw_data_file} to {normalized_data_file}")
