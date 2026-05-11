"""Interactive UMAP visualization of embedded questions and answers.

Loads `data/embedded/*.jsonl` (same schema as `embed_questions.py`), runs UMAP on
stacked question and answer embedding vectors so both kinds share one layout,
and writes an interactive Plotly HTML figure under `data/plots/`.

Marker fill color follows `sequence_number`. Marker outline color follows source
filename. In 2D, question markers use a solid outline and answers a dashed
outline. In 3D, Plotly does not support dashed marker outlines; answers use
diamond markers and questions use circles instead.

Solid segments link each question to its answer when `effective_answer` is
non-empty. Pale dashed segments link consecutive question markers along
`sequence_number` within each `interaction_id`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import umap

_logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

_THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = _THIS_DIR / "data"
EMBEDDED_DIR = DATA_DIR / "embedded"
PLOTS_DIR = DATA_DIR / "plots"


def load_embedded_frames(embedded_dir: Path) -> pd.DataFrame:
    paths = sorted(embedded_dir.glob("*.jsonl"))
    if not paths:
        _logger.warning("No JSONL files under %s", embedded_dir)
        return pd.DataFrame()

    frames = []
    for p in paths:
        chunk = pd.read_json(p, lines=True)
        chunk["_source_file"] = p.name
        frames.append(chunk)
    out = pd.concat(frames, ignore_index=True)
    return out


def _stack_embeddings(df: pd.DataFrame) -> tuple[np.ndarray, list[dict]]:
    """One UMAP row per question embedding, then per answer embedding (paired)."""
    xs: list[np.ndarray] = []
    meta: list[dict] = []

    for pair_index, row in df.iterrows():
        q = np.asarray(row["embedding_question"], dtype=np.float32)
        a = np.asarray(row["embedding_answer"], dtype=np.float32)
        if q.ndim != 1 or a.ndim != 1 or q.shape != a.shape:
            raise ValueError(
                f"Row {pair_index}: expected 1-D question/answer embeddings of equal length, "
                f"got {q.shape!r} and {a.shape!r}"
            )

        has_answer = bool(str(row.get("effective_answer") or "").strip())

        xs.append(q)
        meta.append(
            {
                "pair_index": int(pair_index),
                "kind": "question",
                "has_answer": has_answer,
                "_source_file": row["_source_file"],
                "sequence_number": row["sequence_number"],
                "interaction_id": row["interaction_id"],
                "effective_question": row.get("effective_question", ""),
                "effective_answer": row.get("effective_answer", ""),
            }
        )
        xs.append(a)
        meta.append(
            {
                "pair_index": int(pair_index),
                "kind": "answer",
                "has_answer": has_answer,
                "_source_file": row["_source_file"],
                "sequence_number": row["sequence_number"],
                "interaction_id": row["interaction_id"],
                "effective_question": row.get("effective_question", ""),
                "effective_answer": row.get("effective_answer", ""),
            }
        )

    return np.vstack(xs), meta


def _umap_neighbors(n_samples: int, requested: int) -> int:
    if n_samples < 3:
        return max(2, n_samples - 1)
    return max(2, min(requested, n_samples - 1))


def _line_segments_open(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    *,
    dims: int,
) -> tuple[list | None, list | None, list | None]:
    """Piecewise segments A→B with gaps (None) between pairs."""
    xs: list = []
    ys: list = []
    zs: list | None = [] if dims == 3 else None
    for i in range(len(coords_a)):
        if dims == 2:
            xs.extend([coords_a[i, 0], coords_b[i, 0], None])
            ys.extend([coords_a[i, 1], coords_b[i, 1], None])
        else:
            xs.extend([coords_a[i, 0], coords_b[i, 0], None])
            ys.extend([coords_a[i, 1], coords_b[i, 1], None])
            assert zs is not None
            zs.extend([coords_a[i, 2], coords_b[i, 2], None])
    return xs, ys, zs


def build_figure(
    df: pd.DataFrame,
    *,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> go.Figure:
    if df.empty:
        raise ValueError("No embedded rows to plot.")
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3")

    X, meta = _stack_embeddings(df)
    n_neighbors_eff = _umap_neighbors(len(X), n_neighbors)

    _logger.info(
        "Fitting UMAP: samples=%s dim=%s n_neighbors=%s min_dist=%s metric=%s",
        X.shape[0],
        X.shape[1],
        n_neighbors_eff,
        min_dist,
        metric,
    )
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors_eff,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    coords = reducer.fit_transform(X)

    q_coords: dict[int, np.ndarray] = {}
    a_coords: dict[int, np.ndarray] = {}
    for m, c in zip(meta, coords):
        idx = m["pair_index"]
        if m["kind"] == "question":
            q_coords[idx] = np.asarray(c, dtype=float)
        else:
            a_coords[idx] = np.asarray(c, dtype=float)

    files = sorted(df["_source_file"].unique())
    palette = pc.qualitative.Dark24 + pc.qualitative.Light24
    file_color = {f: palette[i % len(palette)] for i, f in enumerate(files)}

    seq_numeric = pd.to_numeric(df["sequence_number"], errors="coerce")
    cmin = float(np.nanmin(seq_numeric.to_numpy(dtype=float)))
    cmax = float(np.nanmax(seq_numeric.to_numpy(dtype=float)))
    if not np.isfinite(cmin) or not np.isfinite(cmax):
        cmin, cmax = 0.0, 1.0
    if cmax == cmin:
        cmax = cmin + 1.0

    fig = go.Figure()

    # --- Pale dashed paths: consecutive questions within interaction_id ---
    interaction_segments: list[tuple[np.ndarray, np.ndarray]] = []
    for _, g in df.groupby("interaction_id", sort=False):
        seq_sort = pd.to_numeric(g["sequence_number"], errors="coerce")
        g2 = g.assign(_seq_sort=seq_sort).sort_values("_seq_sort")
        pairs = g2.index.tolist()
        for i in range(len(pairs) - 1):
            p0, p1 = int(pairs[i]), int(pairs[i + 1])
            if p0 in q_coords and p1 in q_coords:
                interaction_segments.append((q_coords[p0], q_coords[p1]))

    if interaction_segments:
        ca = np.stack([s[0] for s in interaction_segments], dtype=float)
        cb = np.stack([s[1] for s in interaction_segments], dtype=float)
        lx, ly, lz = _line_segments_open(ca, cb, dims=n_components)
        line_kw: dict = dict(
            color="rgba(160, 168, 190, 0.45)",
            width=1.5,
            dash="dash",
        )
        if n_components == 2:
            fig.add_trace(
                go.Scatter(
                    x=lx,
                    y=ly,
                    mode="lines",
                    line=line_kw,
                    hoverinfo="skip",
                    showlegend=False,
                    name="interaction path",
                )
            )
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=lx,
                    y=ly,
                    z=lz,
                    mode="lines",
                    line=line_kw,
                    hoverinfo="skip",
                    showlegend=False,
                    name="interaction path",
                )
            )

    # --- Solid question → answer connectors ---
    qa_pairs = [
        int(i)
        for i, row in df.iterrows()
        if bool(str(row.get("effective_answer") or "").strip())
        and int(i) in q_coords
        and int(i) in a_coords
    ]
    if qa_pairs:
        ca = np.stack([q_coords[i] for i in qa_pairs])
        cb = np.stack([a_coords[i] for i in qa_pairs])
        lx, ly, lz = _line_segments_open(ca, cb, dims=n_components)
        line_kw = dict(color="rgba(90, 90, 110, 0.85)", width=2)
        if n_components == 2:
            fig.add_trace(
                go.Scatter(
                    x=lx,
                    y=ly,
                    mode="lines",
                    line=line_kw,
                    hoverinfo="skip",
                    showlegend=False,
                    name="Q→A",
                )
            )
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=lx,
                    y=ly,
                    z=lz,
                    mode="lines",
                    line=line_kw,
                    hoverinfo="skip",
                    showlegend=False,
                    name="Q→A",
                )
            )

    Scatter = go.Scatter if n_components == 2 else go.Scatter3d

    def hover_text(kind: str, sub: pd.DataFrame) -> list[str]:
        lines = []
        for _, r in sub.iterrows():
            q_short = str(r.get("effective_question", ""))[:240]
            a_short = str(r.get("effective_answer", ""))[:240]
            lines.append(
                f"<b>{kind}</b><br>"
                f"file: {r['_source_file']}<br>"
                f"interaction_id: {r['interaction_id']}<br>"
                f"sequence_number: {r['sequence_number']}<br>"
                f"<b>Q</b>: {q_short}<br>"
                f"<b>A</b>: {a_short}<extra></extra>"
            )
        return lines

    colorbar_shown = False

    def seq_marker(
        seq_vals: list,
        *,
        fname: str,
        line_dash: str,
        size: float,
        show_colorbar: bool,
        marker_symbol: str = "circle",
    ) -> dict:
        nonlocal colorbar_shown
        line: dict = dict(width=2, color=file_color[fname])
        # Scatter3d marker outlines do not support dash; use symbols to distinguish.
        if n_components == 2:
            line["dash"] = line_dash
        mk: dict = dict(
            size=size,
            color=seq_vals,
            colorscale="Viridis",
            cmin=cmin,
            cmax=cmax,
            symbol=marker_symbol,
            line=line,
        )
        if show_colorbar and not colorbar_shown:
            mk["colorbar"] = dict(title="sequence_number")
            colorbar_shown = True
        return mk

    # Answers first (under questions), then questions on top
    for fname in files:
        sub_a = df[
            (df["_source_file"] == fname)
            & df["effective_answer"].fillna("").astype(str).str.strip().ne("")
        ]
        if len(sub_a) == 0:
            continue
        ax = np.stack([a_coords[int(i)] for i in sub_a.index])
        kwargs = dict(
            mode="markers",
            name=f"{fname} · answers",
            legendgroup=fname,
            showlegend=False,
            marker=seq_marker(
                sub_a["sequence_number"].tolist(),
                fname=fname,
                line_dash="dash",
                size=9,
                show_colorbar=True,
                marker_symbol="diamond" if n_components == 3 else "circle",
            ),
            hovertext=hover_text("answer", sub_a),
            hovertemplate="%{hovertext}",
        )
        if n_components == 2:
            fig.add_trace(Scatter(x=ax[:, 0], y=ax[:, 1], **kwargs))
        else:
            fig.add_trace(Scatter(x=ax[:, 0], y=ax[:, 1], z=ax[:, 2], **kwargs))

    for fname in files:
        sub_q = df[df["_source_file"] == fname]
        qx = np.stack([q_coords[int(i)] for i in sub_q.index])
        kwargs = dict(
            mode="markers",
            name=fname,
            legendgroup=fname,
            showlegend=True,
            marker=seq_marker(
                sub_q["sequence_number"].tolist(),
                fname=fname,
                line_dash="solid",
                size=11,
                show_colorbar=True,
            ),
            hovertext=hover_text("question", sub_q),
            hovertemplate="%{hovertext}",
        )
        if n_components == 2:
            fig.add_trace(Scatter(x=qx[:, 0], y=qx[:, 1], **kwargs))
        else:
            fig.add_trace(Scatter(x=qx[:, 0], y=qx[:, 1], z=qx[:, 2], **kwargs))

    layout_updates = dict(
        template="plotly_white",
        margin=dict(l=0, r=0, t=48, b=0),
        title=dict(text="UMAP of question & answer embeddings"),
        legend=dict(groupclick="toggleitem", tracegroupgap=0),
        hovermode="closest",
    )
    if n_components == 2:
        layout_updates["xaxis"] = dict(title="UMAP 1", zeroline=False)
        layout_updates["yaxis"] = dict(title="UMAP 2", zeroline=False)
        layout_updates["dragmode"] = "pan"
    else:
        layout_updates["scene"] = dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
        )

    fig.update_layout(**layout_updates)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedded-dir",
        type=Path,
        default=EMBEDDED_DIR,
        help="Directory containing embedded JSONL files (default: data/embedded)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PLOTS_DIR / "embedded_umap.html",
        help="Output HTML path (default: data/plots/embedded_umap.html)",
    )
    parser.add_argument(
        "--dims",
        type=int,
        choices=(2, 3),
        default=2,
        help="UMAP target dimensions (default: 2)",
    )
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument(
        "--metric",
        default="cosine",
        help="UMAP distance metric (default: cosine)",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = load_embedded_frames(args.embedded_dir)
    if df.empty:
        raise SystemExit("No data loaded; run embed_questions.py first.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(
        df,
        n_components=args.dims,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
    )
    fig.write_html(args.output, include_plotlyjs="cdn")
    _logger.info("Wrote %s", args.output.resolve())


if __name__ == "__main__":
    main()
