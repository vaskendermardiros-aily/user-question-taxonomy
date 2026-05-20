import json
import re
import textwrap
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import logging

from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer

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
TOPICS_DATA_DIR = DATA_DIR / "topics"
TOPICS_DATA_DIR.mkdir(parents=True, exist_ok=True)

DO_USE_KEY_POINTS = False

_CHAT_ORIGINS = ["free_question", "auto_suggestion", "followup", "auto_followup"]

_CHAT_ORIGIN_PREFIX: dict[str, str] = {
    "free_question": "FQ",  # user typed the question
    "auto_suggestion": "AS",  # user clicked a suggested question
    "followup": "FU",  # user typed a follow-up
    "auto_followup": "AF",  # user clicked a suggested follow-up
}

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Per-user axis overrides for UMAP plots. Users not listed get auto-scaling.
# Fill in after a first run once you can read the actual coordinate ranges.
USER_AXIS_RANGES: dict[str, dict] = {
    # "0f5dec0d-8688-4581-bc0e-5a63b592cf64": {"x_range": (-5, 12), "y_range": (-8, 6)},
}


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

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
    df = df[df["chat_origin"].isin(_CHAT_ORIGINS)]
    df = df.copy()
    df["effective_question"] = df.apply(build_effective_question, axis=1)
    df["effective_answer"] = df.apply(build_effective_answer, axis=1)

    columns_to_keep = [
        "interaction_id",
        "req_user_id",
        "sequence_number",
        "chat_origin",
        "effective_question",
        "effective_answer",
        "req_created_at",
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


def load_all_users() -> pd.DataFrame:
    """Load every raw JSONL file and return a combined interactions DataFrame."""
    raw_data_files = sorted(RAW_DATA_DIR.glob("*.jsonl"))
    if not raw_data_files:
        raise FileNotFoundError(f"No JSONL files found in {RAW_DATA_DIR}")

    frames: list[pd.DataFrame] = []
    for path in raw_data_files:
        _logger.info(f"Loading {path.name}")
        df_raw = pd.read_json(path, lines=True)
        df_clean = clean_input(df_raw)
        df_interactions = format_user_interactions_df(df=df_clean, do_show_answers=True)
        frames.append(df_interactions)

    combined = pd.concat(frames, ignore_index=True)
    combined["req_created_at"] = pd.to_datetime(combined["req_created_at"], utc=True)
    _logger.info(f"Total interactions: {len(combined)} across {combined['req_user_id'].nunique()} users")
    return combined


# ---------------------------------------------------------------------------
# BERTopic model
# ---------------------------------------------------------------------------

def build_topic_model() -> BERTopic:
    """Build a BERTopic model tuned for small corpora (~100-200 docs)."""
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 5D reduction for clustering (reproducible)
    umap_model = UMAP(
        n_neighbors=10,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    # Small min_cluster_size to avoid pushing most docs into the -1 outlier bucket
    hdbscan_model = HDBSCAN(
        min_cluster_size=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # No stop_words: corpus is French/English mixed
    vectorizer_model = CountVectorizer(stop_words=None, min_df=2)

    ctfidf_model = ClassTfidfTransformer()
    representation_model = KeyBERTInspired()

    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        calculate_probabilities=True,
        verbose=True,
    )


def fit_topics(
    topic_model: BERTopic,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Fit BERTopic on the interaction corpus and attach results to df.

    Adds columns: ``topic_id``, ``topic_prob``, ``topic_label``.
    """
    def _assigned_prob(topic_id: int, row_probs) -> float:
        # topic_id == -1 means outlier; BERTopic doesn't allocate a column for it
        if topic_id < 0 or probs.ndim < 2:
            return 0.0
        try:
            return float(row_probs[topic_id])
        except IndexError:
            return 0.0

    docs = df["interaction"].tolist()
    topics, probs = topic_model.fit_transform(docs)

    probs = np.array(probs)  # ensure (n_docs, n_topics) ndarray
    topic_info = topic_model.get_topic_info()
    id_to_label: dict[int, str] = dict(zip(topic_info["Topic"], topic_info["Name"]))

    df = df.copy()
    df["topic_id"] = topics
    df["topic_prob"] = [_assigned_prob(t, p) for t, p in zip(topics, probs)]
    df["topic_label"] = [id_to_label.get(t, f"topic_{t}") for t in topics]

    return df


# ---------------------------------------------------------------------------
# 2D UMAP for visualisation
# ---------------------------------------------------------------------------

def compute_2d_umap(
    topic_model: BERTopic,
    docs: list[str],
) -> tuple[list[float], list[float]]:
    """Project the corpus embeddings to 2D for scatter plotting.

    Uses the same embedding model as BERTopic but a fresh 2D UMAP pass,
    keeping visualization independent from the 5D clustering UMAP.
    """
    # topic_model.embedding_model is a SentenceTransformerBackend wrapper;
    # the underlying SentenceTransformer lives at .embedding_model on that wrapper.
    embedding_model: SentenceTransformer = topic_model.embedding_model.embedding_model
    embeddings = embedding_model.encode(docs, show_progress_bar=True, normalize_embeddings=True)

    umap_2d = UMAP(
        n_neighbors=10,
        n_components=2,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords = umap_2d.fit_transform(embeddings)
    return coords[:, 0].tolist(), coords[:, 1].tolist()


# ---------------------------------------------------------------------------
# Per-user UMAP plot
# ---------------------------------------------------------------------------

def plot_user_umap(
    df: pd.DataFrame,
    user_id: str,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    output_dir: Path = TOPICS_DATA_DIR,
) -> None:
    """Save a 2D UMAP scatter plot for a single user as a standalone HTML file.

    Parameters
    ----------
    df:
        Combined interactions DataFrame with columns ``umap_x``, ``umap_y``,
        ``topic_label``, ``req_user_id``, ``req_created_at``, ``interaction``.
    user_id:
        The ``req_user_id`` to filter on.
    x_range:
        Optional (min, max) to fix the x-axis. Pass ``None`` for auto-scale.
    y_range:
        Optional (min, max) to fix the y-axis. Pass ``None`` for auto-scale.
    output_dir:
        Directory where the HTML file is written.
    """
    user_df = df[df["req_user_id"] == user_id].copy()
    if user_df.empty:
        _logger.warning(f"No interactions for user {user_id}, skipping plot")
        return

    user_df["interaction_preview"] = user_df["interaction"].apply(
        lambda t: textwrap.shorten(t, width=200, placeholder="…")
    )
    user_df["date"] = user_df["req_created_at"].dt.strftime("%Y-%m-%d")

    # Assign grey to outlier topic -1, distinct colours for real topics
    all_topics = sorted(user_df["topic_label"].unique())
    outlier_label = next((lbl for lbl in all_topics if lbl.startswith("-1_")), None)
    non_outlier = [lbl for lbl in all_topics if lbl != outlier_label]
    color_sequence = px.colors.qualitative.Plotly
    color_map = {lbl: color_sequence[i % len(color_sequence)] for i, lbl in enumerate(non_outlier)}
    if outlier_label:
        color_map[outlier_label] = "#BBBBBB"

    fig = px.scatter(
        user_df,
        x="umap_x",
        y="umap_y",
        color="topic_label",
        color_discrete_map=color_map,
        hover_data={
            "topic_label": True,
            "date": True,
            "interaction_preview": True,
            "umap_x": False,
            "umap_y": False,
        },
        title=f"UMAP 2D — {user_id}",
        labels={"topic_label": "Topic", "umap_x": "UMAP-1", "umap_y": "UMAP-2"},
    )
    fig.update_traces(marker=dict(size=8, opacity=0.85))
    fig.update_layout(
        legend_title_text="Topic",
        width=900,
        height=650,
    )
    if x_range is not None:
        fig.update_xaxes(range=list(x_range))
    if y_range is not None:
        fig.update_yaxes(range=list(y_range))

    output_path = output_dir / f"umap_{user_id}.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    _logger.info(f"Saved UMAP plot → {output_path}")


# ---------------------------------------------------------------------------
# Summaries & CSV outputs
# ---------------------------------------------------------------------------

def summarise_per_user(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with one row per (user, topic), with count and %."""
    grouped = (
        df.groupby(["req_user_id", "topic_id", "topic_label"])
        .size()
        .reset_index(name="count")
    )
    grouped["pct"] = grouped.groupby("req_user_id")["count"].transform(
        lambda s: (s / s.sum() * 100).round(1)
    )
    return grouped.sort_values(["req_user_id", "count"], ascending=[True, False])


def summarise_over_time(
    topic_model: BERTopic,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Use BERTopic's built-in topics_over_time and add req_user_id for per-user slicing."""
    docs = df["interaction"].tolist()
    timestamps = df["req_created_at"].tolist()

    tot = topic_model.topics_over_time(docs, timestamps, nr_bins=10)
    # BERTopic returns: Topic, Words, Frequency, Timestamp
    # Re-attach req_user_id by merging on topic assignment
    topic_to_users = (
        df.groupby("topic_id")["req_user_id"]
        .apply(lambda s: s.value_counts().idxmax())
        .reset_index()
        .rename(columns={"req_user_id": "dominant_user"})
    )
    tot = tot.rename(columns={"Topic": "topic_id", "Frequency": "count", "Timestamp": "timestamp"})
    tot = tot.merge(topic_to_users, on="topic_id", how="left")
    tot["year_month"] = pd.to_datetime(tot["timestamp"]).dt.to_period("M").astype(str)
    return tot


def save_outputs(
    df: pd.DataFrame,
    per_user: pd.DataFrame,
    over_time: pd.DataFrame,
) -> None:
    interactions_path = TOPICS_DATA_DIR / "topics_per_interaction.csv"
    per_user_path = TOPICS_DATA_DIR / "topics_per_user.csv"
    over_time_path = TOPICS_DATA_DIR / "topics_over_time.csv"

    cols = ["interaction_id", "req_user_id", "req_created_at", "topic_id", "topic_label", "topic_prob"]
    df[cols].to_csv(interactions_path, index=False)
    per_user.to_csv(per_user_path, index=False)
    over_time.to_csv(over_time_path, index=False)

    _logger.info(f"Saved → {interactions_path}")
    _logger.info(f"Saved → {per_user_path}")
    _logger.info(f"Saved → {over_time_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Load & format all users
    df = load_all_users()

    # 2. Build and fit BERTopic
    topic_model = build_topic_model()
    df = fit_topics(topic_model, df)

    # 3. Compute 2D UMAP coordinates for plotting
    _logger.info("Computing 2D UMAP projection for visualisation…")
    docs = df["interaction"].tolist()
    df["umap_x"], df["umap_y"] = compute_2d_umap(topic_model, docs)

    # 4. Summaries
    per_user_summary = summarise_per_user(df)
    over_time_summary = summarise_over_time(topic_model, df)

    # 5. Print to stdout
    print("\n=== Topic info ===")
    print(topic_model.get_topic_info().to_string(index=False))

    print("\n=== Topics per user ===")
    print(per_user_summary.to_string(index=False))

    print("\n=== Topics over time (all users) ===")
    print(over_time_summary[["topic_id", "Words", "count", "year_month", "dominant_user"]].to_string(index=False))

    # 6. Save CSVs
    save_outputs(df, per_user_summary, over_time_summary)

    # 7. UMAP plots — one per user
    for user_id in df["req_user_id"].unique():
        ranges = USER_AXIS_RANGES.get(user_id, {})
        plot_user_umap(df, user_id, **ranges)

    _logger.info("Done.")
