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
# NORMALIZED_DATA_DIR = DATA_DIR / "normalized"
# NORMALIZED_DATA_DIR.mkdir(parents=True, exist_ok=True)
# USERS_CSV_PATH = DATA_DIR / "users.csv"
DO_USE_KEY_POINTS = False

_CHAT_ORIGINS = ["free_question", "auto_suggestion", "followup", "auto_followup"]

_CHAT_ORIGIN_PREFIX: dict[str, str] = {
    "free_question": "FQ",    # user typed the question
    "auto_suggestion": "AS",  # user clicked a suggested question
    "followup": "FU",          # user typed a follow-up
    "auto_followup": "AF",    # user clicked a suggested follow-up
}

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

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
    if not DO_USE_KEY_POINTS:
        return summary
    key_points = normalize_text(answer.get("key_points", ""))
    return summary + " " + key_points


def clean_input(df: pd.DataFrame) -> pd.DataFrame:
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
        # "req_id",
        # "resp_created_at",
        # "resp_id",
    ]

    return df[columns_to_keep]


def format_user_interactions_df(
    *,
    df: pd.DataFrame,
    do_show_answers: bool = True,
) -> pd.DataFrame:
    """Aggregate per-turn rows into one row per interaction.

    Each interaction's turns are sorted by sequence_number and concatenated
    into a single ``interaction`` string. The prefix for each question is
    derived from ``chat_origin`` (e.g. ``FQ``, ``FU``).

    Legend — question prefixes:
    - FQ: free question (user typed, stronger intent signal)
    - AS: auto-suggestion (user clicked a button, weaker intent signal)
    - FU: follow-up (user typed, strong intent signal)
    - AF: auto follow-up (user clicked a button, weaker intent signal)
    - A:  answer (bot generated)

    Returns a DataFrame with columns:
        - ``interaction_id``
        - ``req_user_id``
        - ``sequence_number`` – max across turns
        - ``req_created_at``  – min across turns (i.e. when the interaction started)
        - ``interaction``     – formatted Q/A text

    Example row (``df.iloc[10]``)::

        interaction_id    396a3db2-54ca-487e-a1b6-1a1581d0c6cd
        req_user_id       0f5dec0d-8688-4581-bc0e-5a63b592cf64
        sequence_number   0
        req_created_at    2026-04-21 13:45:04.314726+00:00
        interaction       FQ: Team performance ABC\nA: Team performa...

    Example ``interaction`` value::

        FQ: Team performance ABC
        A: Team performance for ABC has shown strong growth ...
        FU: Can you expand on metric M?
        A: Metric M is a key performance indicator for the team ...
    """
    rows: list[dict] = []
    for interaction_id, interaction_df in df.groupby("interaction_id"):
        interaction_df = interaction_df.sort_values(by="sequence_number")

        lines: list[str] = []
        for _, turn in interaction_df.iterrows():
            question = str(turn.get("effective_question") or "")
            answer = str(turn.get("effective_answer") or "<none provided>")
            prefix = _CHAT_ORIGIN_PREFIX.get(turn.get("chat_origin"), "Q")
            lines.append(f"{prefix}: {question}")
            if do_show_answers:
                lines.append(f"A: {answer}")

        rows.append({
            "interaction_id": interaction_id,
            "req_user_id": interaction_df["req_user_id"].iloc[0],
            "sequence_number": interaction_df["sequence_number"].max(),
            "req_created_at": interaction_df["req_created_at"].min(),
            "interaction": "\n".join(lines),
        })

    return pd.DataFrame(rows)



def user_pipeline(df: pd.DataFrame):
    df = clean_input(df=df)
    df = format_user_interactions_df(df=df, do_show_answers=True)
    print(df)

if __name__ == "__main__":
    raw_data_files = RAW_DATA_DIR.glob("*.jsonl")
    processed_user = []

    for raw_data_file_path in raw_data_files:
        print(f"Running: {raw_data_file_path}")
        df = pd.read_json(raw_data_file_path, lines=True)
        user_pipeline(df)


"""  BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer


# Step 1 - Extract embeddings
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # example
# NOTE: our data is multilingual, so we use the multilingual model
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Step 2 - Reduce dimensionality
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')

# Step 3 - Cluster reduced embeddings
hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# Step 4 - Tokenize topics
vectorizer_model = CountVectorizer(stop_words="english")

# Step 5 - Create topic representation
ctfidf_model = ClassTfidfTransformer()

# Step 6 - (Optional) Fine-tune topic representations with
# a `bertopic.representation` model
representation_model = KeyBERTInspired()

# All steps together
topic_model = BERTopic(
  embedding_model=embedding_model,          # Step 1 - Extract embeddings
  umap_model=umap_model,                    # Step 2 - Reduce dimensionality
  hdbscan_model=hdbscan_model,              # Step 3 - Cluster reduced embeddings
  vectorizer_model=vectorizer_model,        # Step 4 - Tokenize topics
  ctfidf_model=ctfidf_model,                # Step 5 - Extract topic words
  representation_model=representation_model # Step 6 - (Optional) Fine-tune topic representations
)
"""