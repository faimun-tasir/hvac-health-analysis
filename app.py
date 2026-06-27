import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HVAC Health Analysis | Beximco Pharmaceuticals",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-label { font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-value { font-size: 26px; font-weight: 700; font-family: monospace; margin: 4px 0; }
    .metric-sub { font-size: 11px; color: #6b7280; }
    .section-header {
        font-size: 15px; font-weight: 600; color: #e8eaf0;
        border-left: 3px solid #00d4aa;
        padding-left: 10px; margin: 20px 0 12px 0;
    }
    .capability-box {
        background: #1a1d27; border: 1px solid #2a2d3a;
        border-radius: 8px; padding: 14px 18px; margin-bottom: 10px;
    }
    .event-badge {
        display: inline-block; padding: 3px 10px;
        border-radius: 5px; font-size: 11px; font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background: #1a1d27; border-radius: 6px;
        padding: 8px 18px; color: #6b7280;
    }
    .stTabs [aria-selected="true"] { background: #00d4aa22; color: #00d4aa; }
</style>
""", unsafe_allow_html=True)

# ── DATA LOADING & PROCESSING ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("ahu_data.csv", parse_dates=["datetime"], dayfirst=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    # GMP limits for granulation room (ISO 8 clean room, OSD)
    GMP = {
        "temperature": {"LSL": 18.0, "USL": 25.0},
        "humidity":    {"LSL": 45.0, "USL": 65.0},
        "pressure":    {"LSL": 0.0,  "USL": 30.0},
    }

    # SPC control limits (3-sigma)
    spc = {}
    for col, lim in GMP.items():
        mu = df[col].mean()
        sigma = df[col].std(ddof=1)
        spc[col] = {
            "mean": mu, "std": sigma,
            "UCL": mu + 3*sigma, "LCL": mu - 3*sigma,
            "UWL": mu + 2*sigma, "LWL": mu - 2*sigma,  # warning limits
            "LSL": lim["LSL"],   "USL": lim["USL"],
            "Cp":  (lim["USL"] - lim["LSL"]) / (6*sigma),
            "Cpu": (lim["USL"] - mu) / (3*sigma),
            "Cpl": (mu - lim["LSL"]) / (3*sigma),
            "Cpk": min((lim["USL"] - mu)/(3*sigma), (mu - lim["LSL"])/(3*sigma)),
            "Pp":  (lim["USL"] - lim["LSL"]) / (6*df[col].std()),
            "Ppk": min((lim["USL"] - mu)/(3*df[col].std()), (mu - lim["LSL"])/(3*df[col].std())),
        }

    # Event classification — ordered: most specific first
    def classify(row):
        rh, temp, dp = row["humidity"], row["temperature"], row["pressure"]
        if rh > 70:
            return "🧹 Room Washing / Cleaning"
        if temp > 25.0:
            return "🌡️ Temperature Excursion"
        if temp < 20.0:
            return "❄️ Over-Cooling"
        if dp < 0 and rh > 63:
            return "💧 Negative DP + High RH"
        if dp < 0:
            return "🔴 Negative DP (Door/Filter)"
        if dp > 30:
            return "⚡ DP Spike"
        if rh > 65:
            return "💧 High Humidity"
        return "✅ Normal"

    df["event"] = df.apply(classify, axis=1)
    df["date"]  = df["datetime"].dt.date
    df["hour"]  = df["datetime"].dt.hour
    df["week"]  = df["datetime"].dt.isocalendar().week.astype(int)

    # Outside GMP spec flag
    df["temp_breach"]  = (df["temperature"] < 18) | (df["temperature"] > 25)
    df["rh_breach"]    = (df["humidity"] < 45)    | (df["humidity"] > 65)
    df["dp_breach"]    = df["pressure"] < 0

    # Moving range for control charts
    for col in ["temperature", "humidity", "pressure"]:
        df[f"mr_{col}"] = df[col].diff().abs()

    return df, spc, GMP

df, spc, GMP = load_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏭 Beximco Pharmaceuticals")
    st.markdown("**Unit-03 | FF-075 AHU**")
    st.markdown("**Room:** Granulation (GR-075)")
    st.markdown("**Period:** 24 May – 25 Jun 2026")
    st.markdown("---")
    st.markdown("**Analysis Settings**")

    param_choice = st.selectbox(
        "Select Parameter for SPC Chart",
        ["temperature", "humidity", "pressure"],
        format_func=lambda x: {"temperature": "Temperature (°C)",
                                "humidity": "Humidity (%RH)",
                                "pressure": "Differential Pressure (Pa)"}[x]
    )

    show_washing = st.checkbox("Highlight Room Washing Events", value=True)
    show_limits  = st.checkbox("Show GMP Specification Limits", value=True)
    show_control = st.checkbox("Show SPC Control Limits (3σ)", value=True)
    show_warning = st.checkbox("Show Warning Limits (2σ)", value=False)

    st.markdown("---")
    st.markdown("**GMP Limits Reference**")
    st.markdown("🌡️ Temp: 18 – 25 °C")
    st.markdown("💧 RH: 45 – 65 %RH")
    st.markdown("🔵 DP: > 0 Pa (positive)")
    st.markdown("---")
    st.caption("FF-075 Data Report | SPC Analysis\nIPE Industrial Attachment 2026")

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("## HVAC Health Analysis Using Statistical Process Control")
st.markdown("**AHU FF-075 | Granulation Room | Beximco Pharmaceuticals Unit-03**")
st.markdown("---")

# ── KPI ROW ──────────────────────────────────────────────────────────────────
total = len(df)
breaches = df[df["event"] != "✅ Normal"]
washing  = df[df["event"] == "🧹 Room Washing / Cleaning"]
neg_dp   = df[df["dp_breach"]]
temp_exc = df[df["temp_breach"]]

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Total Readings</div>
        <div class="metric-value" style="color:#00d4aa">{total:,}</div>
        <div class="metric-sub">5-min intervals</div></div>""", unsafe_allow_html=True)
with c2:
    pct = len(breaches)/total*100
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Anomaly Readings</div>
        <div class="metric-value" style="color:#ff4d6d">{len(breaches):,}</div>
        <div class="metric-sub">{pct:.1f}% of total</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Room Washing Events</div>
        <div class="metric-value" style="color:#ffb800">{len(washing):,}</div>
        <div class="metric-sub">High RH &gt; 70%</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Negative DP Events</div>
        <div class="metric-value" style="color:#f97316">{len(neg_dp):,}</div>
        <div class="metric-sub">DP &lt; 0 Pa</div></div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Temp Excursions</div>
        <div class="metric-value" style="color:#a78bfa">{len(temp_exc):,}</div>
        <div class="metric-sub">Outside 18–25°C</div></div>""", unsafe_allow_html=True)
with c6:
    cpk_min = min(spc["temperature"]["Cpk"], spc["humidity"]["Cpk"], spc["pressure"]["Cpk"])
    color = "#00d4aa" if cpk_min >= 1.33 else "#ffb800" if cpk_min >= 1.0 else "#ff4d6d"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Min Cpk (System)</div>
        <div class="metric-value" style="color:{color}">{cpk_min:.3f}</div>
        <div class="metric-sub">Target ≥ 1.33</div></div>""", unsafe_allow_html=True)

st.markdown("")

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 SPC Control Chart",
    "📊 Process Capability",
    "🔍 Event Analysis",
    "📅 Daily Trend",
    "🤖 ML Anomaly Detection",
    "📋 Project Report"
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — SPC CONTROL CHART
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    s = spc[param_choice]
    labels = {"temperature": "Temperature (°C)",
              "humidity":    "Humidity (%RH)",
              "pressure":    "Differential Pressure (Pa)"}
    lbl = labels[param_choice]

    # Downsample every 3 readings for display (15-min)
    df_plot = df.iloc[::3].copy()

    normal  = df_plot[df_plot["event"] == "✅ Normal"]
    washing_pts = df_plot[df_plot["event"] == "🧹 Room Washing / Cleaning"]
    breach_pts  = df_plot[(df_plot["event"] != "✅ Normal") &
                          (df_plot["event"] != "🧹 Room Washing / Cleaning")]

    fig = go.Figure()

    # Normal readings
    fig.add_trace(go.Scatter(
        x=normal["datetime"], y=normal[param_choice],
        mode="lines", name="Normal",
        line=dict(color="#60a5fa", width=1),
        opacity=0.85
    ))

    # Breach readings
    fig.add_trace(go.Scatter(
        x=breach_pts["datetime"], y=breach_pts[param_choice],
        mode="markers", name="Anomaly",
        marker=dict(color="#ff4d6d", size=5, symbol="circle"),
    ))

    # Room washing
    if show_washing and len(washing_pts):
        fig.add_trace(go.Scatter(
            x=washing_pts["datetime"], y=washing_pts[param_choice],
            mode="markers", name="Room Washing",
            marker=dict(color="#ffb800", size=6, symbol="diamond"),
        ))

    # Mean line
    fig.add_hline(y=s["mean"], line_dash="dot", line_color="#00d4aa",
                  annotation_text=f"Mean={s['mean']:.2f}", annotation_position="right")

    # Control limits
    if show_control:
        fig.add_hline(y=s["UCL"], line_dash="dash", line_color="#ff4d6d",
                      annotation_text=f"UCL={s['UCL']:.2f}", annotation_position="right")
        fig.add_hline(y=s["LCL"], line_dash="dash", line_color="#ff4d6d",
                      annotation_text=f"LCL={s['LCL']:.2f}", annotation_position="right")

    # Warning limits
    if show_warning:
        fig.add_hline(y=s["UWL"], line_dash="dot", line_color="#ffb800",
                      annotation_text=f"UWL={s['UWL']:.2f}", annotation_position="right")
        fig.add_hline(y=s["LWL"], line_dash="dot", line_color="#ffb800",
                      annotation_text=f"LWL={s['LWL']:.2f}", annotation_position="right")

    # GMP spec limits
    if show_limits:
        fig.add_hline(y=s["USL"], line_dash="dashdot", line_color="#a78bfa",
                      annotation_text=f"USL={s['USL']}", annotation_position="right")
        fig.add_hline(y=s["LSL"], line_dash="dashdot", line_color="#a78bfa",
                      annotation_text=f"LSL={s['LSL']}", annotation_position="right")

    # Shade room washing periods
    if show_washing:
        wash_dates = df[df["event"] == "🧹 Room Washing / Cleaning"]["date"].unique()
        for d in wash_dates[:20]:  # limit shading
            day_df = df[df["date"] == d]
            wash_df = day_df[day_df["event"] == "🧹 Room Washing / Cleaning"]
            if len(wash_df):
                fig.add_vrect(
                    x0=wash_df["datetime"].min(), x1=wash_df["datetime"].max(),
                    fillcolor="#ffb800", opacity=0.08, line_width=0,
                    annotation_text="wash", annotation_position="top left",
                    annotation_font_size=9
                )

    fig.update_layout(
        title=f"SPC Control Chart — {lbl} | AHU FF-075",
        xaxis_title="Date & Time",
        yaxis_title=lbl,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="#e8eaf0",
        legend=dict(bgcolor="#1a1d27", bordercolor="#2a2d3a", borderwidth=1),
        height=480,
        hovermode="x unified",
        margin=dict(r=120)
    )
    fig.update_xaxes(gridcolor="#2a2d3a", showgrid=True)
    fig.update_yaxes(gridcolor="#2a2d3a", showgrid=True)

    st.plotly_chart(fig, use_container_width=True)

    # SPC stats summary
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Mean", f"{s['mean']:.3f}")
        st.metric("Std Dev (σ)", f"{s['std']:.3f}")
    with col_b:
        st.metric("UCL (μ+3σ)", f"{s['UCL']:.3f}")
        st.metric("LCL (μ-3σ)", f"{s['LCL']:.3f}")
    with col_c:
        st.metric("GMP USL", f"{s['USL']}")
        st.metric("GMP LSL", f"{s['LSL']}")
    with col_d:
        st.metric("Cp", f"{s['Cp']:.3f}")
        st.metric("Cpk", f"{s['Cpk']:.3f}",
                  delta="Capable" if s['Cpk'] >= 1.33 else "Not Capable",
                  delta_color="normal" if s['Cpk'] >= 1.33 else "inverse")

    # Moving range chart
    st.markdown('<div class="section-header">Moving Range Chart (Process Variability)</div>',
                unsafe_allow_html=True)
    mr_col = f"mr_{param_choice}"
    mr_mean = df[mr_col].mean()
    mr_ucl  = 3.267 * mr_mean  # D4 factor for n=2

    fig_mr = go.Figure()
    fig_mr.add_trace(go.Scatter(
        x=df_plot["datetime"], y=df_plot[mr_col],
        mode="lines", name="Moving Range",
        line=dict(color="#34d399", width=1)
    ))
    fig_mr.add_hline(y=mr_mean, line_dash="dot", line_color="#00d4aa",
                     annotation_text=f"MR̄={mr_mean:.3f}")
    fig_mr.add_hline(y=mr_ucl, line_dash="dash", line_color="#ff4d6d",
                     annotation_text=f"UCL={mr_ucl:.3f}")
    fig_mr.update_layout(
        height=220, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="#e8eaf0", margin=dict(r=120, t=10, b=30),
        xaxis=dict(gridcolor="#2a2d3a"), yaxis=dict(gridcolor="#2a2d3a")
    )
    st.plotly_chart(fig_mr, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — PROCESS CAPABILITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.markdown('<div class="section-header">Process Capability Indices — All Parameters</div>',
                unsafe_allow_html=True)

    param_names = {
        "temperature": "Temperature (°C)",
        "humidity":    "Humidity (%RH)",
        "pressure":    "Differential Pressure (Pa)"
    }

    # Capability summary table
    cap_data = []
    for col, name in param_names.items():
        s = spc[col]
        cap_data.append({
            "Parameter": name,
            "Mean": f"{s['mean']:.3f}",
            "Std Dev": f"{s['std']:.3f}",
            "LSL": s["LSL"], "USL": s["USL"],
            "Cp":  f"{s['Cp']:.3f}",
            "Cpl": f"{s['Cpl']:.3f}",
            "Cpu": f"{s['Cpu']:.3f}",
            "Cpk": f"{s['Cpk']:.3f}",
            "Status": "✅ Capable" if s['Cpk'] >= 1.33 else "⚠️ Marginal" if s['Cpk'] >= 1.0 else "❌ Incapable"
        })

    cap_df = pd.DataFrame(cap_data)
    st.dataframe(cap_df, use_container_width=True, hide_index=True)

    st.markdown("""
    **Interpretation Guide:**
    - **Cp** = Process Spread Capability (ignores centering) — measures if process *could* fit within spec
    - **Cpk** = Process Centering Capability — measures if process *actually* fits within spec
    - **Cpl / Cpu** = One-sided indices (lower/upper)
    - **Target:** Cpk ≥ 1.33 (pharmaceutical GMP standard) | Cpk ≥ 1.67 (six-sigma)
    """)

    st.markdown("---")

    # Distribution plots for each parameter
    for col, name in param_names.items():
        s = spc[col]
        st.markdown(f'<div class="section-header">{name} — Process Distribution</div>',
                    unsafe_allow_html=True)

        fig_dist = make_subplots(rows=1, cols=2,
                                  subplot_titles=("Histogram with Normal Fit", "Box Plot by Week"))

        # Histogram
        vals = df[col].dropna()
        hist_vals, bin_edges = np.histogram(vals, bins=60)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        fig_dist.add_trace(go.Bar(
            x=bin_centers, y=hist_vals, name="Observed",
            marker_color="#60a5fa", opacity=0.7
        ), row=1, col=1)

        # Normal fit
        x_fit = np.linspace(vals.min(), vals.max(), 200)
        y_fit = stats.norm.pdf(x_fit, s["mean"], s["std"]) * len(vals) * (bin_edges[1]-bin_edges[0])
        fig_dist.add_trace(go.Scatter(
            x=x_fit, y=y_fit, name="Normal Fit",
            line=dict(color="#00d4aa", width=2)
        ), row=1, col=1)

        # Spec & control lines
        for y_val, color, dash, lbl in [
            (s["USL"], "#a78bfa", "dashdot", f"USL={s['USL']}"),
            (s["LSL"], "#a78bfa", "dashdot", f"LSL={s['LSL']}"),
            (s["UCL"], "#ff4d6d", "dash",    f"UCL={s['UCL']:.2f}"),
            (s["LCL"], "#ff4d6d", "dash",    f"LCL={s['LCL']:.2f}"),
            (s["mean"], "#00d4aa", "dot",    f"μ={s['mean']:.2f}"),
        ]:
            fig_dist.add_vline(x=y_val, line_dash=dash, line_color=color,
                               annotation_text=lbl, annotation_font_size=9,
                               row=1, col=1)

        # Box plot by week
        df_box = df.copy()
        df_box["Week"] = "Wk " + df_box["week"].astype(str)
        for week in sorted(df_box["week"].unique()):
            wk_data = df_box[df_box["week"] == week][col]
            fig_dist.add_trace(go.Box(
                y=wk_data, name=f"Wk {week}",
                marker_color="#60a5fa", line_color="#00d4aa",
                showlegend=False
            ), row=1, col=2)

        fig_dist.update_layout(
            height=320, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="#e8eaf0", showlegend=False,
            margin=dict(t=40, b=30)
        )
        fig_dist.update_xaxes(gridcolor="#2a2d3a")
        fig_dist.update_yaxes(gridcolor="#2a2d3a")
        st.plotly_chart(fig_dist, use_container_width=True)

        # Capability metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Cp",  f"{s['Cp']:.3f}",  help="Potential capability (ignores centering)")
        c2.metric("Cpk", f"{s['Cpk']:.3f}", help="Actual capability (centering considered)")
        c3.metric("Cpl", f"{s['Cpl']:.3f}", help="Lower-side capability")
        c4.metric("Cpu", f"{s['Cpu']:.3f}", help="Upper-side capability")
        c5.metric("Pp",  f"{s['Pp']:.3f}",  help="Performance index (long-term)")
        st.markdown("---")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — EVENT ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown('<div class="section-header">Event Classification — Root Cause Analysis</div>',
                unsafe_allow_html=True)

    event_counts = df["event"].value_counts()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        # Pie chart
        colors_map = {
            "✅ Normal":                    "#00d4aa",
            "🧹 Room Washing / Cleaning":   "#ffb800",
            "🔴 Negative DP (Door/Filter)": "#ff4d6d",
            "💧 Negative DP + High RH":     "#f97316",
            "⚡ DP Spike":                  "#a78bfa",
            "🌡️ Temperature Excursion":     "#60a5fa",
            "❄️ Over-Cooling":              "#34d399",
            "💧 High Humidity":             "#fb923c",
        }
        colors = [colors_map.get(e, "#6b7280") for e in event_counts.index]

        fig_pie = go.Figure(go.Pie(
            labels=event_counts.index,
            values=event_counts.values,
            marker=dict(colors=colors),
            hole=0.4,
            textinfo="percent+label",
            textfont_size=11
        ))
        fig_pie.update_layout(
            title="Reading Distribution by Event Type",
            height=380, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="#e8eaf0", showlegend=False
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        # Event frequency bar
        non_normal = event_counts[event_counts.index != "✅ Normal"]
        fig_bar = go.Figure(go.Bar(
            x=non_normal.values,
            y=non_normal.index,
            orientation="h",
            marker_color=["#ffb800","#ff4d6d","#f97316","#a78bfa","#60a5fa","#34d399","#fb923c"][:len(non_normal)],
            text=non_normal.values, textposition="outside"
        ))
        fig_bar.update_layout(
            title="Anomaly Event Frequency",
            height=380, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="#e8eaf0",
            xaxis=dict(gridcolor="#2a2d3a", title="Number of Readings"),
            yaxis=dict(gridcolor="#2a2d3a"),
            margin=dict(l=220)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Root cause explanation
    st.markdown('<div class="section-header">Root Cause Interpretation</div>',
                unsafe_allow_html=True)

    causes = {
        "🧹 Room Washing / Cleaning": {
            "trigger": "RH > 70%RH",
            "cause": "Clean room sanitization with water/disinfectant causes temporary humidity spike. AHU dehumidification takes 30–90 min to recover.",
            "gmp_impact": "Expected event — not a GMP violation if within scheduled cleaning windows. Document in logbook.",
            "action": "Exclude washing periods from Cpk calculation. Log start/end time of cleaning.",
            "color": "#ffb800"
        },
        "🔴 Negative DP (Door/Filter)": {
            "trigger": "DP < 0 Pa",
            "cause": "Pressure differential reversal — most likely caused by door opening, filter blockage, or AHU fan issue. Granulation room should maintain positive pressure to prevent cross-contamination.",
            "gmp_impact": "HIGH RISK — negative DP allows unfiltered air ingress. Potential contamination event.",
            "action": "Immediate investigation. Check door seals, HEPA filter differential pressure, supply/exhaust balance.",
            "color": "#ff4d6d"
        },
        "⚡ DP Spike": {
            "trigger": "DP > 30 Pa",
            "cause": "Sudden pressure surge — may indicate blocked return air duct, damper malfunction, or fan speed surge.",
            "gmp_impact": "Moderate risk — excessive pressure may stress filter media and door seals.",
            "action": "Check supply/return air balance. Inspect damper positions and VAV controllers.",
            "color": "#a78bfa"
        },
        "🌡️ Temperature Excursion": {
            "trigger": "Temperature > 25°C",
            "cause": "Chiller capacity issue, high ambient load (equipment heat generation), or AHU cooling coil performance degradation.",
            "gmp_impact": "Moderate — may affect granulation process and product stability.",
            "action": "Check chilled water supply temperature, AHU cooling coil, and room heat load inventory.",
            "color": "#60a5fa"
        },
    }

    for event, info in causes.items():
        count = event_counts.get(event, 0)
        if count > 0:
            with st.expander(f"{event} — {count} readings ({count/total*100:.1f}%)", expanded=False):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"**Trigger Condition:** `{info['trigger']}`")
                    st.markdown(f"**Root Cause:** {info['cause']}")
                with c2:
                    st.markdown(f"**GMP Impact:** {info['gmp_impact']}")
                with c3:
                    st.markdown(f"**Recommended Action:** {info['action']}")

    # Hourly heatmap of anomalies
    st.markdown('<div class="section-header">Anomaly Heatmap — Hour of Day vs Date</div>',
                unsafe_allow_html=True)

    df["is_anomaly"] = (df["event"] != "✅ Normal").astype(int)
    heatmap_data = df.groupby(["date", "hour"])["is_anomaly"].sum().reset_index()
    heatmap_pivot = heatmap_data.pivot(index="hour", columns="date", values="is_anomaly").fillna(0)

    fig_heat = go.Figure(go.Heatmap(
        z=heatmap_pivot.values,
        x=[str(c) for c in heatmap_pivot.columns],
        y=[f"{h:02d}:00" for h in heatmap_pivot.index],
        colorscale="RdYlGn_r",
        colorbar=dict(title="Anomaly Count")
    ))
    fig_heat.update_layout(
        height=380, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="#e8eaf0",
        xaxis=dict(title="Date", tickangle=-45),
        yaxis=dict(title="Hour of Day"),
        margin=dict(b=80)
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — DAILY TREND
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    daily = df.groupby("date").agg(
        avg_temp=("temperature", "mean"),
        avg_rh=("humidity", "mean"),
        avg_dp=("pressure", "mean"),
        max_temp=("temperature", "max"),
        max_rh=("humidity", "max"),
        min_dp=("pressure", "min"),
        anomaly_count=("is_anomaly", "sum"),
        total_count=("temperature", "count")
    ).reset_index()
    daily["anomaly_pct"] = daily["anomaly_count"] / daily["total_count"] * 100
    daily["date_str"] = daily["date"].astype(str)

    st.markdown('<div class="section-header">Daily Average Trends — All Parameters</div>',
                unsafe_allow_html=True)

    fig_daily = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=(
            "Temperature (°C) — Daily Average",
            "Humidity (%RH) — Daily Average",
            "Differential Pressure (Pa) — Daily Average",
            "Daily Anomaly Count"
        ),
        vertical_spacing=0.06
    )

    # Temperature
    fig_daily.add_trace(go.Scatter(
        x=daily["date_str"], y=daily["avg_temp"], name="Avg Temp",
        line=dict(color="#60a5fa", width=2), fill="tozeroy", fillcolor="rgba(96,165,250,0.1)"
    ), row=1, col=1)
    fig_daily.add_hline(y=25, line_dash="dash", line_color="#ff4d6d",
                         annotation_text="USL=25°C", row=1, col=1)
    fig_daily.add_hline(y=18, line_dash="dash", line_color="#ff4d6d",
                         annotation_text="LSL=18°C", row=1, col=1)

    # Humidity
    fig_daily.add_trace(go.Scatter(
        x=daily["date_str"], y=daily["avg_rh"], name="Avg RH",
        line=dict(color="#a78bfa", width=2), fill="tozeroy", fillcolor="rgba(167,139,250,0.1)"
    ), row=2, col=1)
    fig_daily.add_hline(y=65, line_dash="dash", line_color="#ff4d6d",
                         annotation_text="USL=65%", row=2, col=1)
    fig_daily.add_hline(y=45, line_dash="dash", line_color="#ff4d6d",
                         annotation_text="LSL=45%", row=2, col=1)

    # Pressure
    fig_daily.add_trace(go.Scatter(
        x=daily["date_str"], y=daily["avg_dp"], name="Avg DP",
        line=dict(color="#34d399", width=2)
    ), row=3, col=1)
    fig_daily.add_hline(y=0, line_dash="dash", line_color="#ff4d6d",
                         annotation_text="LSL=0 Pa", row=3, col=1)

    # Anomaly count
    fig_daily.add_trace(go.Bar(
        x=daily["date_str"], y=daily["anomaly_count"], name="Anomalies",
        marker_color="#ff4d6d", opacity=0.8
    ), row=4, col=1)

    fig_daily.update_layout(
        height=700, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="#e8eaf0", showlegend=False,
        margin=dict(r=100)
    )
    for i in range(1, 5):
        fig_daily.update_xaxes(gridcolor="#2a2d3a", row=i, col=1)
        fig_daily.update_yaxes(gridcolor="#2a2d3a", row=i, col=1)

    st.plotly_chart(fig_daily, use_container_width=True)

    # Daily data table
    st.markdown('<div class="section-header">Daily Summary Table</div>',
                unsafe_allow_html=True)
    display_daily = daily[["date_str","avg_temp","avg_rh","avg_dp","max_temp","max_rh","min_dp","anomaly_count","anomaly_pct"]].copy()
    display_daily.columns = ["Date","Avg Temp°C","Avg RH%","Avg DP Pa","Max Temp","Max RH","Min DP","Anomalies","Anomaly%"]
    display_daily = display_daily.round(2)
    st.dataframe(display_daily, use_container_width=True, hide_index=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — ML ANOMALY DETECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab5:
    st.markdown('<div class="section-header">Isolation Forest — Unsupervised Anomaly Detection</div>',
                unsafe_allow_html=True)
    st.info("ℹ️ Isolation Forest learns *normal* operating patterns and flags readings that deviate — without needing labeled failure data. Complements SPC by detecting multivariate anomalies invisible to single-parameter control charts.")

    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    features = ["temperature", "humidity", "pressure"]
    X = df[features].dropna()
    idx = X.index

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    contamination = st.slider(
        "Expected Anomaly Rate (contamination parameter)",
        min_value=0.01, max_value=0.20, value=0.05, step=0.01,
        help="Set to roughly the fraction of readings you expect to be anomalous"
    )

    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    preds = model.fit_predict(X_scaled)
    scores = model.score_samples(X_scaled)

    df_ml = df.loc[idx].copy()
    df_ml["if_pred"]  = preds   # -1 = anomaly, 1 = normal
    df_ml["if_score"] = scores  # more negative = more anomalous

    n_anomalies = (df_ml["if_pred"] == -1).sum()
    st.metric("Isolation Forest Anomalies Detected",
              f"{n_anomalies:,}",
              f"{n_anomalies/len(df_ml)*100:.1f}% of readings")

    # Anomaly score time series
    df_ml_plot = df_ml.iloc[::3]
    fig_if = go.Figure()
    normal_if  = df_ml_plot[df_ml_plot["if_pred"] == 1]
    anomaly_if = df_ml_plot[df_ml_plot["if_pred"] == -1]

    fig_if.add_trace(go.Scatter(
        x=normal_if["datetime"], y=normal_if["if_score"],
        mode="lines", name="Normal", line=dict(color="#00d4aa", width=1)
    ))
    fig_if.add_trace(go.Scatter(
        x=anomaly_if["datetime"], y=anomaly_if["if_score"],
        mode="markers", name="ML Anomaly",
        marker=dict(color="#ff4d6d", size=5)
    ))
    fig_if.update_layout(
        title="Isolation Forest Anomaly Score Over Time",
        xaxis_title="Date", yaxis_title="Anomaly Score (lower = more anomalous)",
        height=340, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="#e8eaf0",
        xaxis=dict(gridcolor="#2a2d3a"), yaxis=dict(gridcolor="#2a2d3a")
    )
    st.plotly_chart(fig_if, use_container_width=True)

    # 3D scatter — feature space
    st.markdown('<div class="section-header">3D Feature Space — Normal vs Anomaly</div>',
                unsafe_allow_html=True)
    df_3d = df_ml.sample(min(1500, len(df_ml)), random_state=42)
    fig_3d = go.Figure()
    for label, color, name in [(1, "#00d4aa", "Normal"), (-1, "#ff4d6d", "ML Anomaly")]:
        sub = df_3d[df_3d["if_pred"] == label]
        fig_3d.add_trace(go.Scatter3d(
            x=sub["temperature"], y=sub["humidity"], z=sub["pressure"],
            mode="markers", name=name,
            marker=dict(size=2.5, color=color, opacity=0.6)
        ))
    fig_3d.update_layout(
        height=420, paper_bgcolor="#0e1117", font_color="#e8eaf0",
        scene=dict(
            xaxis=dict(title="Temperature°C", backgroundcolor="#0e1117", gridcolor="#2a2d3a"),
            yaxis=dict(title="Humidity%RH",   backgroundcolor="#0e1117", gridcolor="#2a2d3a"),
            zaxis=dict(title="Pressure Pa",   backgroundcolor="#0e1117", gridcolor="#2a2d3a"),
        )
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    # Agreement between SPC and ML
    st.markdown('<div class="section-header">SPC vs ML Agreement Analysis</div>',
                unsafe_allow_html=True)
    df_ml["spc_anomaly"] = (df_ml["event"] != "✅ Normal").astype(int)
    df_ml["ml_anomaly"]  = (df_ml["if_pred"] == -1).astype(int)
    df_ml["agree"]       = df_ml["spc_anomaly"] == df_ml["ml_anomaly"]

    agree_rate = df_ml["agree"].mean() * 100
    both_flag  = ((df_ml["spc_anomaly"]==1) & (df_ml["ml_anomaly"]==1)).sum()
    spc_only   = ((df_ml["spc_anomaly"]==1) & (df_ml["ml_anomaly"]==0)).sum()
    ml_only    = ((df_ml["spc_anomaly"]==0) & (df_ml["ml_anomaly"]==1)).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Agreement Rate", f"{agree_rate:.1f}%")
    c2.metric("Both SPC & ML Flag", f"{both_flag:,}", help="High-confidence anomalies")
    c3.metric("SPC Only", f"{spc_only:,}", help="Rule-based detections missed by ML")
    c4.metric("ML Only", f"{ml_only:,}", help="Multivariate patterns missed by SPC rules")

    st.markdown("""
    **Interpretation:** When both SPC and Isolation Forest flag the same reading,
    it is a **high-confidence anomaly** — two independent methods agree. ML-only flags 
    may represent subtle multivariate patterns not captured by single-parameter SPC rules.
    SPC-only flags (e.g., room washing) are rule-based events the ML model may normalize 
    if they occur frequently.
    """)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 6 — PROJECT REPORT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab6:
    st.markdown("## Project Report")
    st.markdown("### HVAC Health Analysis Using Statistical Process Control")
    st.markdown("**Beximco Pharmaceuticals Ltd. | Unit-03 | AHU FF-075 | Industrial Attachment 2026**")
    st.markdown("---")

    st.markdown("""
    #### 1. Introduction
    Pharmaceutical manufacturing facilities must maintain strict environmental conditions
    in clean rooms to comply with GMP (Good Manufacturing Practice) guidelines under
    WHO, EU GMP Annex 1, and Bangladesh DGDA regulations. The granulation room (FF-075)
    at Beximco Unit-03 is an ISO 8 classified area where temperature, relative humidity,
    and differential pressure must be continuously monitored and controlled.

    Current practice relies on threshold alarms that fire only after a parameter has
    already breached its limit. This project applies **Statistical Process Control (SPC)**
    techniques combined with **Machine Learning anomaly detection** to provide:
    - Early warning before limit breaches
    - Quantitative process capability assessment (Cp, Cpk)
    - Root cause classification of environmental excursions
    - A data-driven framework for predictive maintenance
    """)

    st.markdown("""
    #### 2. Objectives
    1. Parse and clean one month of real AHU environmental monitoring data (FF-075)
    2. Compute SPC control charts (X̄, MR) for Temperature, RH, and Differential Pressure
    3. Calculate process capability indices: Cp, Cpk, Cpl, Cpu, Pp, Ppk
    4. Classify excursion events with root cause reasoning (room washing, negative DP, etc.)
    5. Apply Isolation Forest ML to detect multivariate anomalies
    6. Identify peak anomaly periods and operational patterns
    7. Provide GMP-actionable recommendations for PES department
    """)

    st.markdown("""
    #### 3. Data Description
    | Field | Details |
    |-------|---------|
    | Source | FF-075 EMS Data Report, Beximco Unit-03 |
    | Period | 24 May 2026 – 25 Jun 2026 (33 days) |
    | Interval | 5-minute readings |
    | Total Records | 9,217 readings |
    | Parameters | Temperature (°C), Humidity (%RH), DP_R075-R077 (Pa) |
    | Missing Values | None (complete dataset) |
    """)

    st.markdown("#### 4. Process Capability Results")
    results_data = {
        "Parameter": ["Temperature (°C)", "Humidity (%RH)", "Differential Pressure (Pa)"],
        "Mean": [23.42, 60.57, 10.66],
        "Std Dev": [0.697, 4.736, 9.114],
        "LSL": [18.0, 45.0, 0.0],
        "USL": [25.0, 65.0, 30.0],
        "Cp":  [1.675, 0.704, 0.549],
        "Cpk": [0.757, 0.312, 0.390],
        "Verdict": ["⚠️ Off-center", "❌ Incapable", "❌ Incapable"]
    }
    st.dataframe(pd.DataFrame(results_data), use_container_width=True, hide_index=True)

    st.markdown("""
    **Key Finding:** While temperature Cp (1.675) indicates the process *could* fit within
    the 18–25°C specification, Cpk (0.757) reveals significant off-centering toward the
    upper limit. Humidity and pressure show Cpk < 1.0, indicating incapable processes
    with frequent excursions — largely attributable to room washing events and door opening.

    #### 5. Event Classification Summary
    | Event Type | Readings | % | Root Cause |
    |---|---|---|---|
    | Normal Operation | 7,040 | 76.4% | — |
    | Negative DP | 1,120 | 12.1% | Door opening / filter blockage |
    | Negative DP + High RH | 438 | 4.8% | Washing with door open |
    | Room Washing | 362 | 3.9% | Scheduled clean room sanitization |
    | DP Spike | 217 | 2.4% | Damper/fan transient |
    | Temperature Excursion | 40 | 0.4% | Chiller/heat load issue |

    #### 6. Conclusions
    1. **Temperature** is well-controlled in average but off-centered — mean (23.42°C) sits
       close to the 25°C upper limit, reducing Cpk to 0.757. Recommend setpoint reduction.
    2. **Humidity** shows Cpk of 0.312 — driven primarily by room washing events. Excluding
       washing periods, Cpk improves significantly. Document cleaning schedules in EMS.
    3. **Differential Pressure** shows the highest anomaly rate (17%) due to frequent negative
       DP events. Door management and filter maintenance are the primary corrective actions.
    4. **ML detection** agreed with SPC in the majority of cases, and identified additional
       multivariate anomaly patterns invisible to single-parameter control charts.

    #### 7. Recommendations
    1. Reduce HVAC temperature setpoint to 22°C to improve Cpk centering
    2. Implement door interlock alarms to reduce negative DP frequency
    3. Schedule filter replacement based on DP trend (currently irregular)
    4. Tag EMS data with room cleaning timestamps to separate process capability
       from scheduled maintenance events
    5. Expand this analysis to all AHU units in OSD3/SSL2 for facility-wide benchmarking
    """)

    st.markdown("---")
    st.caption("Prepared by: IPE Industrial Attachment Student | Beximco Pharmaceuticals PES Dept. | June 2026")
