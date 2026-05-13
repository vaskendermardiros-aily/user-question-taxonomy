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

Click a marker to dim everything except that `interaction_id` (path + points);
click the same marker again to clear. Clicking empty plot area clears when the
browser sends a click with no picked point.

Hover tooltips list file, ids, timestamps, then Q/A, using only Plotly-supported
markup (e.g. ``<b>``, ``<br>``); long fields use ``textwrap`` because ``<div>`` etc.
are not rendered as HTML in Plotly hovers. Translucent hover panels use injected
CSS (``layout.hoverlabel`` rgba is often ignored by plotly.js for ``hovermode: closest``).
"""

from __future__ import annotations

import argparse
import html
import logging
import textwrap
from collections import defaultdict
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

# Plotly hovers only treat a small HTML subset as markup (e.g. <br>, <b>, <i>).
# Wider tags like <div>/<span> are shown as literal text — use textwrap + <br> only.
_HOVER_TEXT_WRAP_WIDTH = 42
# Plotly often ignores alpha in layout.hoverlabel.bgcolor for hovermode "closest" (and
# 3D); see https://stackoverflow.com/questions/67386595/plotly-hoverlabel-color-transparency
# Real translucency is applied via scoped CSS in _HOVER_TRANSPARENCY_POST_SCRIPT (fill-opacity
# on the hover background path). Range ~0.25–0.55; text is forced fully opaque.
_HOVER_BOX_PATH_FILL_OPACITY = 0.42

def _wrap_hover_plain(text: object, *, wrap_width: int | None = None) -> str:
    """Plain text -> HTML-safe fragment using only <br> (Plotly-supported in hovers)."""
    w = _HOVER_TEXT_WRAP_WIDTH if wrap_width is None else wrap_width
    raw = str(text or "").strip()
    if not raw:
        return html.escape("(empty)")
    lines_out: list[str] = []
    for para in raw.split("\n"):
        if not para:
            lines_out.append("")
            continue
        wrapped = textwrap.wrap(
            para,
            width=w,
            break_long_words=True,
            replace_whitespace=False,
        )
        lines_out.extend(wrapped if wrapped else [""])
    return "<br>".join(html.escape(line) for line in lines_out)


def _hover_meta_line(label: str, value: object, *, wrap: bool = False) -> str:
    """One labeled row; Plotly-safe (<b>, <br> only in markup)."""
    lab = html.escape(str(label))
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raw = ""
    else:
        raw = str(value).strip()
    if not raw:
        body = html.escape("(empty)")
    elif wrap:
        body = _wrap_hover_plain(raw)
    else:
        body = html.escape(raw)
    return f"<b>{lab}</b><br>{body}"


def _marker_hovertext(sub: pd.DataFrame) -> list[str]:
    """Hover: metadata + Q/A; only <b>/<br> markup (Plotly hovers)."""
    out: list[str] = []
    for _, r in sub.iterrows():
        meta_parts = [
            _hover_meta_line("file", r.get("_source_file")),
            _hover_meta_line("req_user_id", r.get("req_user_id")),
            _hover_meta_line("interaction_id", r.get("interaction_id"), wrap=True),
            _hover_meta_line("sequence_number", r.get("sequence_number")),
        ]
        if pd.notna(r.get("chat_origin")):
            meta_parts.append(_hover_meta_line("chat_origin", r.get("chat_origin")))
        for col, lab in (
            ("req_created_at", "req_created_at"),
            ("resp_created_at", "resp_created_at"),
            ("req_id", "req_id"),
            ("resp_id", "resp_id"),
        ):
            if col in r.index and pd.notna(r.get(col)):
                meta_parts.append(_hover_meta_line(lab, r.get(col), wrap=True))
        meta_block = "<br>".join(meta_parts)
        q = _wrap_hover_plain(r.get("effective_question"))
        a = _wrap_hover_plain(r.get("effective_answer"))
        out.append(f"{meta_block}<br><br><b>Q</b><br>{q}<br><br><b>A</b><br>{a}")
    return out


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


def _polyline_coords(
    segments: list[tuple[np.ndarray, np.ndarray]],
    dims: int,
) -> tuple[list, list, list | None]:
    """Open polyline through segment endpoints with None breaks between segments."""
    xs: list = []
    ys: list = []
    zs: list | None = [] if dims == 3 else None
    for ca, cb in segments:
        if dims == 2:
            xs.extend([float(ca[0]), float(cb[0]), None])
            ys.extend([float(ca[1]), float(cb[1]), None])
        else:
            xs.extend([float(ca[0]), float(cb[0]), None])
            ys.extend([float(ca[1]), float(cb[1]), None])
            assert zs is not None
            zs.extend([float(ca[2]), float(cb[2]), None])
    return xs, ys, zs


# Appended after Plotly.newPlot; `{plot_id}` is replaced by write_html before injection.
_CLICK_HIGHLIGHT_POST_SCRIPT = """
(function () {
  var gd = document.getElementById('{plot_id}');
  if (!gd || !window.Plotly) return;

  var selectedIid = null;

  function iidFromPoint(pt) {
    if (!pt || pt.customdata === undefined || pt.customdata === null) return null;
    var v = pt.customdata;
    if (Array.isArray(v)) v = v[0];
    return String(v);
  }

  function isMarkerTrace(tr) {
    return tr && tr.mode && tr.mode.indexOf('markers') !== -1 && tr.customdata;
  }

  function applyHighlight(activeIid) {
    var markerIdx = [];
    var markerOp = [];
    var lineIdx = [];
    var lineOp = [];
    for (var t = 0; t < gd.data.length; t++) {
      var tr = gd.data[t];
      if (isMarkerTrace(tr)) {
        var n = tr.x.length;
        var op = new Array(n);
        for (var i = 0; i < n; i++) {
          var cid = tr.customdata[i];
          if (Array.isArray(cid)) cid = cid[0];
          op[i] = String(cid) === activeIid ? 1.0 : 0.12;
        }
        markerIdx.push(t);
        markerOp.push(op);
      } else if (tr.meta && tr.meta.interaction_id !== undefined && tr.meta.default_opacity !== undefined) {
        var match = String(tr.meta.interaction_id) === activeIid;
        lineIdx.push(t);
        lineOp.push(match ? tr.meta.default_opacity : 0.06);
      }
    }
    if (markerIdx.length) Plotly.restyle(gd, { 'marker.opacity': markerOp }, markerIdx);
    if (lineIdx.length) Plotly.restyle(gd, { 'opacity': lineOp }, lineIdx);
  }

  function resetHighlight() {
    var markerIdx = [];
    var markerOp = [];
    var lineIdx = [];
    var lineOp = [];
    for (var t = 0; t < gd.data.length; t++) {
      var tr = gd.data[t];
      if (isMarkerTrace(tr)) {
        var n = tr.x.length;
        var op = new Array(n);
        for (var i = 0; i < n; i++) op[i] = 1.0;
        markerIdx.push(t);
        markerOp.push(op);
      } else if (tr.meta && typeof tr.meta.default_opacity === 'number') {
        lineIdx.push(t);
        lineOp.push(tr.meta.default_opacity);
      }
    }
    if (markerIdx.length) Plotly.restyle(gd, { 'marker.opacity': markerOp }, markerIdx);
    if (lineIdx.length) Plotly.restyle(gd, { 'opacity': lineOp }, lineIdx);
  }

  gd.on('plotly_click', function (ev) {
    if (!ev.points || !ev.points.length) {
      selectedIid = null;
      resetHighlight();
      return;
    }
    var iid = iidFromPoint(ev.points[0]);
    if (iid === null) {
      selectedIid = null;
      resetHighlight();
      return;
    }
    if (selectedIid === iid) {
      selectedIid = null;
      resetHighlight();
      return;
    }
    selectedIid = iid;
    applyHighlight(iid);
  });
})();
"""

# Scoped CSS: layout hoverlabel rgba is often ignored by plotly.js for closest-hover / 3D.
_HOVER_TRANSPARENCY_POST_SCRIPT = (
    """
(function () {
  var plotId = '{plot_id}';
  var css =
    '#' + plotId + ' g.hovertext > path { fill-opacity: __UMAP_FILL_OP__ !important; }\\n' +
    '#' + plotId + ' g.hovertext text, #' + plotId + ' g.hovertext tspan { opacity: 1 !important; fill-opacity: 1 !important; }';
  var st = document.createElement('style');
  st.setAttribute('data-embedded-umap-hover-transparency', '1');
  st.textContent = css;
  document.head.appendChild(st);
})();
""".replace("__UMAP_FILL_OP__", str(_HOVER_BOX_PATH_FILL_OPACITY))
)


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

    path_default_opacity = 0.45
    qa_default_opacity = 0.85

    # One file per interaction (for legendgroup so lines hide with marker traces).
    iid_to_source_file = {
        str(k): str(v) for k, v in df.groupby("interaction_id", sort=False)["_source_file"].first().items()
    }

    # --- Pale dashed paths: one trace per interaction_id (for hover dimming) ---
    segments_by_iid: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for iid, g in df.groupby("interaction_id", sort=False):
        seq_sort = pd.to_numeric(g["sequence_number"], errors="coerce")
        g2 = g.assign(_seq_sort=seq_sort).sort_values("_seq_sort")
        pairs = g2.index.tolist()
        for i in range(len(pairs) - 1):
            p0, p1 = int(pairs[i]), int(pairs[i + 1])
            if p0 in q_coords and p1 in q_coords:
                segments_by_iid[str(iid)].append((q_coords[p0], q_coords[p1]))

    line_kw_path: dict = dict(
        color="rgba(160, 168, 190, 0.45)",
        width=1.5,
        dash="dash",
    )
    for iid, segs in segments_by_iid.items():
        if not segs:
            continue
        lx, ly, lz = _polyline_coords(segs, n_components)
        src_file = iid_to_source_file.get(str(iid), "")
        meta = dict(
            trace_type="interaction_path",
            interaction_id=iid,
            default_opacity=path_default_opacity,
        )
        if n_components == 2:
            fig.add_trace(
                go.Scatter(
                    x=lx,
                    y=ly,
                    mode="lines",
                    line=line_kw_path,
                    opacity=path_default_opacity,
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=src_file,
                    name="interaction path",
                    meta=meta,
                )
            )
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=lx,
                    y=ly,
                    z=lz,
                    mode="lines",
                    line=dict(color="rgba(160, 168, 190, 0.45)", width=1.5, dash="dash"),
                    opacity=path_default_opacity,
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=src_file,
                    name="interaction path",
                    meta=meta,
                )
            )

    # --- Solid question → answer: one trace per row (for hover dimming) ---
    qa_pairs = [
        int(i)
        for i, row in df.iterrows()
        if bool(str(row.get("effective_answer") or "").strip())
        and int(i) in q_coords
        and int(i) in a_coords
    ]
    line_kw_qa = dict(color="rgba(90, 90, 110, 0.85)", width=2)
    for idx in qa_pairs:
        row = df.loc[idx]
        iid = str(row["interaction_id"])
        src_file = str(row["_source_file"])
        q_ = q_coords[idx]
        a_ = a_coords[idx]
        meta = dict(
            trace_type="qa_connector",
            interaction_id=iid,
            default_opacity=qa_default_opacity,
        )
        if n_components == 2:
            fig.add_trace(
                go.Scatter(
                    x=[float(q_[0]), float(a_[0])],
                    y=[float(q_[1]), float(a_[1])],
                    mode="lines",
                    line=line_kw_qa,
                    opacity=qa_default_opacity,
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=src_file,
                    name="Q→A",
                    meta=meta,
                )
            )
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=[float(q_[0]), float(a_[0])],
                    y=[float(q_[1]), float(a_[1])],
                    z=[float(q_[2]), float(a_[2])],
                    mode="lines",
                    line=dict(color="rgba(90, 90, 110, 0.85)", width=2),
                    opacity=qa_default_opacity,
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=src_file,
                    name="Q→A",
                    meta=meta,
                )
            )

    Scatter = go.Scatter if n_components == 2 else go.Scatter3d

    def seq_marker(
        seq_vals: list,
        *,
        fname: str,
        line_dash: str,
        size: float,
        marker_symbol: str = "circle",
    ) -> dict:
        line: dict = dict(width=2, color=file_color[fname])
        # Scatter3d marker outlines do not support dash; use symbols to distinguish.
        if n_components == 2:
            line["dash"] = line_dash
        return dict(
            size=size,
            color=seq_vals,
            colorscale="Viridis",
            cmin=cmin,
            cmax=cmax,
            symbol=marker_symbol,
            line=line,
            showscale=False,
        )

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
            customdata=sub_a["interaction_id"].astype(str).tolist(),
            marker=seq_marker(
                sub_a["sequence_number"].tolist(),
                fname=fname,
                line_dash="dash",
                size=9,
                marker_symbol="diamond" if n_components == 3 else "circle",
            ),
            hovertext=_marker_hovertext(sub_a),
            hovertemplate="%{hovertext}<extra></extra>",
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
            customdata=sub_q["interaction_id"].astype(str).tolist(),
            marker=seq_marker(
                sub_q["sequence_number"].tolist(),
                fname=fname,
                line_dash="solid",
                size=11,
            ),
            hovertext=_marker_hovertext(sub_q),
            hovertemplate="%{hovertext}<extra></extra>",
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
        hoverlabel=dict(
            bgcolor="rgb(255, 255, 255)",
            bordercolor="rgba(55, 55, 75, 0.38)",
            align="left",
            font=dict(
                size=12,
                family="system-ui, -apple-system, Segoe UI, sans-serif",
                color="rgb(24, 24, 34)",
            ),
        ),
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
    fig.write_html(
        args.output,
        include_plotlyjs="cdn",
        div_id="embedded-umap-plot",
        post_script=[_CLICK_HIGHLIGHT_POST_SCRIPT, _HOVER_TRANSPARENCY_POST_SCRIPT],
    )
    _logger.info("Wrote %s", args.output.resolve())


if __name__ == "__main__":
    main()
