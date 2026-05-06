"""Create reduced embeddings for normalized questions.

This script reads `data/unique_questions_normalized.csv`, encodes each
`effective_question` with a multilingual sentence-transformer model, reduces
the embeddings with PCA, and writes a compact dataset that can be reused by
different clustering scripts.

The output is written to `data/unique_questions_reduced_embeddings.csv`.
"""

import csv
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 128
REDUCED_DIMENSIONS = 10


def load_rows(input_path: Path) -> list[dict[str, str]]:
    with input_path.open("r", newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def encode_questions(
    model: SentenceTransformer,
    questions: list[str],
):
    return model.encode(
        questions,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def reduce_embeddings(embeddings):
    reducer = PCA(n_components=REDUCED_DIMENSIONS)
    return reducer.fit_transform(embeddings)


def serialize_embedding(values) -> str:
    rounded_values = [round(float(value), 6) for value in values]
    return json.dumps(rounded_values, ensure_ascii=False)


def process_file(input_path: Path, output_path: Path) -> None:
    rows = load_rows(input_path)
    questions = [row["effective_question"] for row in rows]

    print(f"Loaded rows: {len(rows)}")
    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = encode_questions(model, questions)
    reduced_embeddings = reduce_embeddings(embeddings)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "effective_question",
        "created_at",
        "dim_reduced_embedding",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for row, reduced_embedding in zip(rows, reduced_embeddings):
            writer.writerow(
                {
                    "effective_question": row["effective_question"],
                    "created_at": row["created_at"],
                    "dim_reduced_embedding": serialize_embedding(reduced_embedding),
                }
            )

    print(f"Reduced dimensions: {REDUCED_DIMENSIONS}")
    print(f"Saved rows: {len(rows)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    input_path = Path("data/unique_questions_normalized.csv")
    output_path = Path("data/unique_questions_reduced_embeddings.csv")
    process_file(input_path, output_path)
