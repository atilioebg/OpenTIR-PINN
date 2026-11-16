"""
Dataset Validation Report tab for the OpenTIR-PINN Spectral & Dataset Explorer.
Renders data/validation/dataset_validation_report.json (produced by scripts/validate_dataset_integrity.py):
every test result is shown with its obtained value next to its pass/fail limit.
"""
import os
import json

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    go = None

STATUS_COLOR = {"PASS": "#10b981", "WARN": "#f59e0b", "FAIL": "#ef4444", "INFO": "#64748b"}
STATUS_ICON = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "INFO": "ℹ️"}
OPS = {"<=": "≤", "<": "<", ">=": "≥", ">": ">", "==": "=", "info": "—"}


def _pipeline_root():
    """Folder containing run_pipeline.py: the repository root if present, otherwise the release kit."""
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, ".."), os.path.join(here, "pipeline"), here):
        if os.path.exists(os.path.join(p, "run_pipeline.py")):
            return os.path.abspath(p)
    return None


def _locate_report():
    """Most recent validation report among the published copy and the local pipeline run."""
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.path.join(here, "data", "validation", "dataset_validation_report.json"),
             os.path.join(here, "..", "data", "validation", "dataset_validation_report.json"),
             os.path.join(here, "pipeline", "data", "validation", "dataset_validation_report.json")]
    found = [os.path.abspath(p) for p in cands if os.path.exists(p)]
    return max(found, key=os.path.getmtime) if found else None


# -----------------------------------------------------------------------------------------------
# Pipeline runner (button -> background subprocess -> live log -> refreshed report)
# -----------------------------------------------------------------------------------------------
RUN_MODES = {
    "Validation only (manifest + all tests, ~4 min; needs existing tensors)": ["--stages", "facts", "validate"],
    "Full pipeline from raw data (~5 min)": [],
    "Full pipeline + MODTRAN LUT replication (~11 min, licensed MODTRAN 5.2)": ["--run-modtran"],
}
STAGE_NAMES = ["setup", "atmosphere", "curate", "split", "synthesize", "facts", "validate", "reproduce"]


def _modtran_available():
    d = os.environ.get("OPENTIR_MODTRAN_DIR", r"C:\PcModWin5\bin")
    return os.path.exists(os.path.join(d, os.environ.get("OPENTIR_MODTRAN_EXE", "Mod5.2.0.0.exe")))


def _lock_path(root):
    return os.path.join(root, "logs", ".pipeline_running")


def _start_run(root, mode, n_null):
    import subprocess, sys, time
    os.makedirs(os.path.join(root, "logs"), exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(root, "logs", f"dashboard_run_{stamp}.log")
    cmd = [sys.executable, os.path.join(root, "run_pipeline.py"), *RUN_MODES[mode], "--n-null", str(n_null)]
    fp = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=root, stdout=fp, stderr=subprocess.STDOUT,
                            env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1"),
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    with open(_lock_path(root), "w") as lf:
        lf.write(json.dumps({"pid": proc.pid, "started": stamp, "mode": mode}))
    st.session_state["val_run"] = {"proc": proc, "log": log_path, "fp": fp, "mode": mode, "t0": time.time(), "root": root}


def _tail(path, n=25):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()[-n:]
    except OSError:
        return []


def _progress(lines_all, mode):
    stages = RUN_MODES[mode]
    expected = ["facts", "validate"] if "--stages" in stages else STAGE_NAMES
    done = [s for s in expected if any(f": {s} ===" in l for l in lines_all)]
    return len(done), len(expected), (done[-1] if done else "starting")


@st.fragment(run_every="3s")
def _live_panel():
    import time
    run = st.session_state.get("val_run")
    if not run:
        return
    proc = run["proc"]
    lines_all = _tail(run["log"], 100000)
    k, n, cur = _progress(lines_all, run["mode"])
    elapsed = time.time() - run["t0"]
    if proc.poll() is None:
        st.progress(min(k / n, 0.99), text=f"Running stage {k}/{n}: **{cur}** · {elapsed / 60:.1f} min elapsed")
        st.code("\n".join(lines_all[-18:]) or "starting ...", language="text")
        return
    # finished
    run["fp"].close()
    try:
        os.remove(_lock_path(run["root"]))
    except OSError:
        pass
    st.session_state["val_last"] = {"rc": proc.returncode, "log": run["log"], "minutes": elapsed / 60, "mode": run["mode"]}
    del st.session_state["val_run"]
    _load.clear()
    st.rerun(scope="app")


def _render_runner():
    import time
    root = _pipeline_root()
    with st.container(border=True):
        st.markdown("##### ▶️ Run the pipeline and the validation protocol")
        if root is None:
            st.info("`run_pipeline.py` not found next to the dashboard; the runner is unavailable.")
            return
        c1, c2, c3 = st.columns([2.4, 1.0, 0.9])
        modes = list(RUN_MODES)
        if not _modtran_available():
            modes = modes[:2]
        has_tensors = os.path.exists(os.path.join(root, "data", "processed", "final_datasets", "train_fold0_aster.pt"))
        if not has_tensors:
            # clean working folder: "Validation only" would fail (it re-checks existing tensors)
            modes = [m for m in modes if not m.startswith("Validation only")]
            if st.session_state.get("val_mode") not in modes:
                st.session_state.pop("val_mode", None)
            st.info("No generated tensors in `data/processed/final_datasets/` (clean working folder). "
                    "**Validation only** is hidden until a full run has generated them: choose **Full pipeline from raw data**.")
        mode = c1.selectbox("Mode", modes, key="val_mode",
                            help="Validation only re-checks the current tensors. Full rebuilds everything from the raw "
                                 "inputs and compares with REFERENCE_CHECKSUMS.json (release kit). MODTRAN replication "
                                 "re-executes the 330 published decks (shown only if MODTRAN 5.2 is installed).")
        n_null = c2.slider("Null size (C7b)", 10, 50, 30, 5, key="val_nnull",
                           help="Random group-stratified splits used by test C7b; lower is faster.")
        running = "val_run" in st.session_state
        lock = _lock_path(root)
        foreign = (not running and os.path.exists(lock) and time.time() - os.path.getmtime(lock) < 45 * 60)
        clicked = c3.button("▶️ Run", type="primary", disabled=running or foreign, key="val_run_btn",
                            use_container_width=True)
        rel_root = os.path.relpath(root, os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
        st.caption(f"Working folder: `{rel_root}/` (relative to the dataset folder) · outputs are written to its `data/`, `splits/` and `logs/` folders "
                   f"(about 1.3 GB for a full run).")
        if foreign:
            st.warning("Another pipeline run is in progress (lock file present). Wait for it to finish.")
        if clicked and not running and not foreign:
            _start_run(root, mode, n_null)
            st.rerun()
        if "val_run" in st.session_state:
            _live_panel()
        last = st.session_state.get("val_last")
        if last:
            ok = last["rc"] == 0
            (st.success if ok else st.error)(
                f"{'✅ Pipeline SUCCESS' if ok else '❌ Pipeline FAILED'} ({last['mode'].split(' (')[0]}, "
                f"{last['minutes']:.1f} min). The report below has been reloaded.")
            with st.expander("Execution log (last 40 lines)", expanded=not ok):
                st.code("\n".join(_tail(last["log"], 40)), language="text")
        summ = os.path.join(root, "logs", "pipeline_summary.json")
        if os.path.exists(summ):
            with st.expander("Last pipeline run: stage summary", expanded=False):
                s = json.load(open(summ, encoding="utf-8"))
                st.caption(f"Result: **{s.get('result')}** · started {s.get('started')} · finished {s.get('finished')}")
                st.dataframe(pd.DataFrame([{"Stage": k, "Status": v["status"], "Seconds": v["seconds"],
                                            "Error": v.get("error", "")} for k, v in s["stages"].items()]),
                             hide_index=True, width="stretch")
                rc = os.path.join(root, "logs", "reproducibility_check.json")
                if os.path.exists(rc):
                    st.caption("Reproducibility vs published release: " + " · ".join(
                        f"{'✅' if v else '❌'} {k}" for k, v in json.load(open(rc)).items()))
                lr = os.path.join(root, "data", "validation", "lut_replication", "lut_replication_report.csv")
                if os.path.exists(lr):
                    d = pd.read_csv(lr)
                    st.caption(f"MODTRAN LUT replication: {int(d.passed.sum())}/{len(d)} checks within tolerance, "
                               f"max relative difference {d.max_rel_diff.max():.2e}")


@st.cache_data(show_spinner=False)
def _load(path, mtime):
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _margin(v, op, lim):
    if v is None or lim is None or op == "info":
        return np.nan
    if op in ("<=", "<"):
        return lim - v
    if op in (">=", ">"):
        return v - lim
    return -abs(v - lim)


def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if float(v).is_integer() and abs(v) < 1e7:
        return f"{int(v)}"
    return f"{v:.3g}" if (abs(v) < 1e-3 or abs(v) >= 1e4) else f"{v:.4f}"


def _flatten(report):
    rows = []
    for r in report["results"]:
        for m in r["metrics"]:
            ok = None
            if m["op"] != "info" and m["limit"] is not None:
                mg = _margin(m["value"], m["op"], m["limit"])
                ok = (mg >= 0) if m["op"] in ("<=", ">=") else ((mg > 0) if m["op"] in ("<", ">") else abs(mg) <= 1e-12)
            rows.append({"Status": r["status"], "ID": r["id"], "Section": r["section"], "Test": r["title"],
                         "Sensor": r["sensor"], "Scope": r["scope"], "Metric": m["name"],
                         "Obtained": m["value"], "Rule": OPS[m["op"]], "Limit": m["limit"],
                         "Level": m["level"] if m["op"] != "info" else "INFO",
                         "Within limit": ("—" if ok is None else ("yes" if ok else "NO")),
                         "Margin": _margin(m["value"], m["op"], m["limit"]), "Detail": r["detail"]})
    return pd.DataFrame(rows)


def _glance_cards(df):
    """Worst case per (test, metric) across sensors/scopes: obtained vs limit."""
    judged = df[df["Rule"] != "—"].copy()
    if judged.empty:
        return
    judged["rank"] = judged["Margin"].fillna(np.inf)
    worst = judged.sort_values("rank").groupby(["ID", "Metric"], as_index=False, sort=False).first()
    worst = worst.rename(columns={"Within limit": "Within"})
    worst["order"] = worst["ID"].map(lambda s: (s[0], s))
    worst = worst.sort_values("order")
    cols = st.columns(4)
    for i, r in enumerate(worst.itertuples()):
        status = "PASS" if r.Within == "yes" else ("FAIL" if r.Level == "FAIL" else "WARN")
        c = STATUS_COLOR[status]
        cols[i % 4].markdown(
            f'<div style="background:#0f172a;border:1px solid #334155;border-left:5px solid {c};border-radius:8px;'
            f'padding:8px 10px;margin-bottom:8px;">'
            f'<div style="font-size:0.70rem;color:#94a3b8;font-weight:700;">{STATUS_ICON[status]} {r.ID} · {r.Test[:44]}</div>'
            f'<div style="font-size:0.72rem;color:#cbd5e1;margin:2px 0 4px 0;font-family:monospace;">{r.Metric}</div>'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
            f'<span style="font-size:1.05rem;font-weight:700;color:{c};">{_fmt(r.Obtained)}</span>'
            f'<span style="font-size:0.80rem;color:#e2e8f0;">limit {r.Rule} {_fmt(r.Limit)}</span></div>'
            f'<div style="font-size:0.66rem;color:#64748b;">worst case: {r.Sensor} · {r.Scope}</div></div>',
            unsafe_allow_html=True)


def _drilldown(df):
    judged = df[df["Rule"] != "—"]
    if judged.empty or go is None:
        return
    c1, c2, c3 = st.columns([1.2, 1.6, 0.8])
    tid = c1.selectbox("Test", sorted(judged["ID"].unique(), key=lambda s: (s[0], s)), key="val_tid")
    sub = judged[judged["ID"] == tid]
    met = c2.selectbox("Metric", sorted(sub["Metric"].unique()), key="val_met")
    sub = sub[sub["Metric"] == met].copy()
    sub["x"] = sub["Sensor"] + " · " + sub["Scope"]
    vals = sub["Obtained"].astype(float).values
    lim = float(sub["Limit"].iloc[0])
    op = sub["Rule"].iloc[0]
    positive = np.all(vals > 0) and lim > 0
    logy = c3.checkbox("Log scale", value=bool(positive and (vals.max() / max(vals.min(), 1e-300) > 100 or lim / max(vals.min(), 1e-300) > 100)),
                       disabled=not positive, key="val_log")
    colors = [STATUS_COLOR["PASS"] if w == "yes" else (STATUS_COLOR["FAIL"] if lv == "FAIL" else STATUS_COLOR["WARN"])
              for w, lv in zip(sub["Within limit"], sub["Level"])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["x"], y=vals, mode="markers+text", marker=dict(size=13, color=colors, line=dict(width=1, color="#e2e8f0")),
                             text=[_fmt(v) for v in vals], textposition="top center", name="obtained",
                             hovertemplate="%{x}<br>obtained = %{y:.6g}<extra></extra>"))
    fig.add_hline(y=lim, line=dict(color="#f8fafc", width=2, dash="dash"),
                  annotation_text=f"limit {op} {_fmt(lim)}", annotation_position="top left",
                  annotation_font=dict(color="#f8fafc"))
    lo = min(vals.min(), lim); hi = max(vals.max(), lim)
    pad = (hi - lo) * 0.25 if hi > lo else (abs(hi) * 0.25 + 1e-9)
    if op in ("≤", "<"):
        pass_band, fail_band = (lo - pad, lim), (lim, hi + pad)
    elif op in ("≥", ">"):
        pass_band, fail_band = (lim, hi + pad), (lo - pad, lim)
    else:
        pass_band, fail_band = None, None
    if pass_band and not logy:
        fig.add_hrect(y0=pass_band[0], y1=pass_band[1], fillcolor=STATUS_COLOR["PASS"], opacity=0.10, line_width=0)
        fig.add_hrect(y0=fail_band[0], y1=fail_band[1], fillcolor=STATUS_COLOR["FAIL"], opacity=0.10, line_width=0)
    fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title=met, xaxis_title="sensor · scope", showlegend=False,
                      yaxis_type="log" if logy else "linear")
    st.plotly_chart(fig, width="stretch")
    st.caption("Green band = pass region, red band = fail region, dashed line = limit. Marker colour = status of each scope.")


def render_validation_tab():
    st.subheader("✅ Dataset Validation Report: Obtained Values vs. Pass/Fail Limits")
    _render_runner()
    path = _locate_report()
    if path is None:
        st.warning("Validation report not found. Run `python scripts/validate_dataset_integrity.py`.")
        return
    rep = _load(path, os.path.getmtime(path))
    df = _flatten(rep)
    s = rep.get("summary", {})
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("✅ PASS", s.get("PASS", 0))
    k2.metric("⚠️ WARN", s.get("WARN", 0))
    k3.metric("❌ FAIL", s.get("FAIL", 0))
    k4.metric("ℹ️ INFO", s.get("INFO", 0))
    verdict = "RELEASE-READY" if s.get("FAIL", 0) == 0 else "BLOCKED"
    k5.metric("Release gate", verdict)
    st.caption(f"Dataset v{rep.get('dataset_version', '?')} · generated {rep.get('generated_at', '?')} · "
               f"NEdT = {rep['constants'].get('nedt_K_by_sensor', rep['constants']['nedt_K'])} K · spectral-twin radius RMSE = {rep['constants']['spectral_twin_rmse']} "
               f"· source: `{os.path.relpath(path)}`. FAIL blocks release; each WARN needs a documented PI decision.")

    st.markdown("##### 🎯 At a glance: worst case of every judged metric (obtained vs. limit)")
    _glance_cards(df)

    st.markdown("##### 🔎 Drill-down: obtained value per sensor/fold against the limit")
    _drilldown(df)

    st.markdown("##### 📋 Full results table")
    f1, f2, f3, f4 = st.columns(4)
    secs = f1.multiselect("Section", sorted(df["Section"].unique()), default=sorted(df["Section"].unique()),
                          format_func=lambda x: f"{x}. {rep.get('sections', {}).get(x, '')}")
    sens = f2.multiselect("Sensor", sorted(df["Sensor"].unique()), default=sorted(df["Sensor"].unique()))
    stats_ = f3.multiselect("Status", ["FAIL", "WARN", "PASS", "INFO"], default=["FAIL", "WARN", "PASS", "INFO"])
    q = f4.text_input("Search (ID, test, metric)", "")
    view = df[df["Section"].isin(secs) & df["Sensor"].isin(sens) & df["Status"].isin(stats_)]
    if q:
        ql = q.lower()
        view = view[view["ID"].str.lower().str.contains(ql) | view["Test"].str.lower().str.contains(ql)
                    | view["Metric"].str.lower().str.contains(ql)]
    show = view[["Status", "ID", "Test", "Sensor", "Scope", "Metric", "Obtained", "Rule", "Limit", "Within limit", "Margin", "Level"]].copy()
    show["Status"] = show["Status"].map(lambda x: f"{STATUS_ICON[x]} {x}")

    def _style(row):
        st_ = row["Status"].split()[-1]
        c = STATUS_COLOR[st_]
        styles = [""] * len(row)
        styles[0] = f"color:{c};font-weight:700"
        styles[6] = f"color:{c if row['Within limit'] != '—' else '#cbd5e1'};font-weight:700"
        styles[9] = f"color:{STATUS_COLOR['PASS'] if row['Within limit'] == 'yes' else (c if row['Within limit'] == 'NO' else '#64748b')}"
        return styles

    st.dataframe(show.style.apply(_style, axis=1).format({"Obtained": _fmt, "Limit": _fmt, "Margin": _fmt}),
                 width="stretch", hide_index=True, height=480)
    st.download_button("📥 Download results (CSV)", view.to_csv(index=False).encode("utf-8"),
                       file_name="dataset_validation_results.csv", mime="text/csv")

    with st.expander("📖 Test protocol: purpose, method, pass rule and rationale of every test", expanded=False):
        cat = pd.DataFrame([{"ID": k, "Section": v["section"], "Test": v["name"], "Purpose": v["purpose"],
                             "Method": v["method"], "Pass rule": v["rule"], "Rationale": v["rationale"]}
                            for k, v in rep.get("catalog", {}).items()])
        st.dataframe(cat, width="stretch", hide_index=True)
        st.caption("Full protocol: `docs/dataset_validation_protocol.md` (auto-generated from the validator's TEST_CATALOG).")

    with st.expander("🧪 Adversarial null distributions (C7b)", expanded=False):
        if go is not None and rep.get("adversarial_null"):
            fig = go.Figure()
            for sensor, null in rep["adversarial_null"].items():
                fig.add_trace(go.Histogram(x=null, name=f"null {sensor}", opacity=0.55, nbinsx=30))
                obs = df[(df["ID"] == "C7b") & (df["Sensor"] == sensor)]
                for r in obs.itertuples():
                    fig.add_vline(x=r.Obtained, line=dict(color=STATUS_COLOR[r.Status], width=2),
                                  annotation_text=f"{sensor} {r.Scope}", annotation_font=dict(size=9))
            fig.update_layout(template="plotly_dark", barmode="overlay", height=340, xaxis_title="adversarial ROC-AUC",
                              margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Histogram = AUC of random group-stratified splits (expected shift of any group-disjoint CV). "
                       "Vertical lines = actual folds; a fold passes if it lies below the 95th percentile.")
