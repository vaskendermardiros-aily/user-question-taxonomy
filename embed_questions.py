"""Create reduced embeddings for normalized questions and answers.

Reads `data/normalized/*.jsonl` (see `normalize_questions.py`), encodes
`effective_question` and `effective_answer` with a multilingual
sentence-transformer model, reduces each with PCA, and writes JSONL under
`data/embedded/` with the same basename as each input file.
"""

import json
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

import logging

_logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Paths
_THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = _THIS_DIR / "data"
NORMALIZED_DATA_DIR = DATA_DIR / "normalized"
EMBEDDED_DATA_DIR = DATA_DIR / "embedded"
EMBEDDED_DATA_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 128
REDUCED_DIMENSIONS = 10


def _text_series(df: pd.DataFrame, column: str) -> list[str]:
    return df[column].fillna("").astype(str).str.strip().replace("", " ").tolist()


def encode_texts(model: SentenceTransformer, texts: list[str]):
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def reduce_embeddings(embeddings):
    reducer = PCA(n_components=REDUCED_DIMENSIONS)
    return reducer.fit_transform(embeddings)


def round_embedding(values) -> list[float]:
    return [round(float(value), 9) for value in values]


def _json_scalar(value):
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def process_file(input_path: Path, output_path: Path, model: SentenceTransformer) -> None:
    df = pd.read_json(input_path, lines=True)

    if len(df) == 0:
        _logger.warning(f"No rows in {input_path}")
        return

    questions = _text_series(df, "effective_question")
    answers = _text_series(df, "effective_answer")

    _logger.info(f"Loaded rows: {len(df)} from {input_path}")
    _logger.info(f"Embedding model: {EMBEDDING_MODEL_NAME}")

    q_emb = encode_texts(model, questions)
    a_emb = encode_texts(model, answers)

    df["embedding_question"] = [round_embedding(row) for row in q_emb]
    df["embedding_answer"] = [round_embedding(row) for row in a_emb]

    q_reduced = reduce_embeddings(q_emb)
    a_reduced = reduce_embeddings(a_emb)

    records = df.to_dict(orient="records")
    for row, q_row, a_row in zip(records, q_reduced, a_reduced):
        row["embedding_question_reduced"] = round_embedding(q_row)
        row["embedding_answer_reduced"] = round_embedding(a_row)

    with output_path.open("w", encoding="utf-8") as f:
        for row in records:
            safe = {k: _json_scalar(v) if not isinstance(v, list) else v for k, v in row.items()}
            f.write(json.dumps(safe, default=str, ensure_ascii=False) + "\n")

    _logger.info(f"Reduced dimensions (each): {REDUCED_DIMENSIONS}")
    _logger.info(f"Saved rows: {len(records)}")
    _logger.info(f"Output: {output_path}")


if __name__ == "__main__":
    normalized_data_files = sorted(NORMALIZED_DATA_DIR.glob("*.jsonl"))

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    for normalized_data_file_path  in normalized_data_files:
        normalized_data_file_name = normalized_data_file_path.stem
        embedded_data_file_path = EMBEDDED_DATA_DIR / f"{normalized_data_file_name}.jsonl"

        _logger.info(f"Processing {embedded_data_file_path}...")
        process_file(normalized_data_file_path, embedded_data_file_path, model)
        _logger.info(f"Processed {normalized_data_file_path} to {embedded_data_file_path}")
