"""
app.py — Cost-Aware Aerial Defence System
Minimalistic Light-Themed Single-Page Interface (White, Grey, Black).
"""

import sys
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parent
_SRC  = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.data_loader import load_config
from src.pipeline import run_pipeline


# ─── Page config ───
st.set_page_config(
    page_title="Cost-Aware Aerial Defence System",
    page_icon="🛡️",
    layout="wide",
)

# ─── Minimalist Light Theme (White, Grey, Black) Custom CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --bg-main: #ffffff;
    --bg-card: #f8f9fa;
    --bg-card-hover: #f1f5f9;
    --border-subtle: #e2e8f0;
    --border-strong: #94a3b8;
    --text-black: #0f172a;
    --text-dark: #1e293b;
    --text-muted: #64748b;
    --text-dim: #94a3b8;
    --accent-black: #000000;
}

.stApp {
    background-color: var(--bg-main) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: var(--text-dark) !important;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

section[data-testid="stSidebar"] {
    display: none !important;
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-black) !important;
    letter-spacing: -0.02em;
}

h1 {
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    text-transform: uppercase;
    color: #000000 !important;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 14px 18px;
    transition: border-color 0.2s ease;
}

div[data-testid="stMetric"]:hover {
    border-color: var(--border-strong);
}

div[data-testid="stMetric"] label {
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text-black) !important;
    font-weight: 800 !important;
    font-size: 1.6rem !important;
}

/* Inputs & Selectbox Dropdown */
.stSelectbox > div > div,
.stTextInput > div > div > input {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 6px !important;
}

.stSelectbox > div > div:hover,
.stTextInput > div > div > input:hover {
    border-color: var(--border-strong) !important;
}

/* Style Dropdown Options Menu */
ul[data-testid="stSelectboxVirtualDropdown"],
div[data-baseweb="popover"],
div[data-baseweb="menu"] {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
}

div[data-baseweb="option"] {
    background-color: #ffffff !important;
    color: #0f172a !important;
}

div[data-baseweb="option"]:hover,
div[aria-selected="true"] {
    background-color: #f1f5f9 !important;
    color: #000000 !important;
}

/* Buttons */
.stButton > button {
    background: #000000 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    padding: 10px 28px !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background: #334155 !important;
    color: #ffffff !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15) !important;
}

/* Header Box */
.header-light {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 32px 36px 28px;
    margin-bottom: 28px;
    text-align: center;
}

.header-light h1 {
    font-size: 2.1rem !important;
    margin-bottom: 8px !important;
}

.header-light p {
    color: var(--text-muted);
    font-size: 0.95rem;
    margin: 0;
    line-height: 1.5;
}

/* Minimalist Cards */
.mini-card-light {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 22px 26px;
    margin: 14px 0;
}

.mini-card-light h4 {
    margin: 0 0 10px 0 !important;
    font-size: 0.8rem !important;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
}

.mini-card-light .val {
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--text-black);
}

.mini-card-light .sub {
    font-size: 0.88rem;
    color: var(--text-muted);
    margin-top: 6px;
    line-height: 1.6;
}

/* Recommendation Box */
.rec-box-light {
    background: #ffffff;
    border: 2px solid #000000;
    border-radius: 10px;
    padding: 24px 28px;
    margin: 16px 0;
}

/* Decision Band Pill Badges (Light Monochrome) */
.badge-light {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    background: #f1f5f9;
    color: #0f172a;
    border: 1px solid #cbd5e1;
}

/* Justification Box */
.justification-light {
    background: var(--bg-card);
    border-left: 3px solid #000000;
    border-radius: 0 8px 8px 0;
    padding: 18px 22px;
    margin: 16px 0;
    color: var(--text-dark);
    font-size: 0.88rem;
    line-height: 1.65;
}

/* Warning Box */
.warning-light {
    background: #fef2f2;
    border-left: 3px solid #64748b;
    border-radius: 0 8px 8px 0;
    padding: 18px 22px;
    margin: 16px 0;
    color: var(--text-dark);
}

/* Table styling (Light) */
.stDataFrame {
    background: #ffffff !important;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border-subtle);
}

/* Plotly container */
.stPlotlyChart {
    background: #ffffff !important;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border-subtle);
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-main); }
::-webkit-scrollbar-thumb { background: var(--border-subtle); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-strong); }
</style>
""", unsafe_allow_html=True)


# ─── Light Monochrome Plotly Theme ───
LIGHT_PLOTLY = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8f9fa",
    font=dict(family="Inter", color="#475569", size=12),
    margin=dict(l=40, r=20, t=45, b=40),
    xaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1"),
    yaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1"),
)


# ─── Data Pipeline Loader (Cached) ───
@st.cache_data(show_spinner="Initializing Defence System Pipeline...")
def load_data():
    results, threats, cms, feasible = run_pipeline()
    return results, threats, cms, feasible


# ─── Main Application ───
def main():
    # ── Header ──
    st.markdown("""
    <div class="header-light">
        <h1>COST AWARE AERIAL DEFENCE SYSTEM</h1>
        <p>Threat priority scoring, hard constraint feasibility matching, and cost-aware decision engine</p>
    </div>
    """, unsafe_allow_html=True)

    results, threats, cms, feasible = load_data()
    cfg = load_config()
    de = cfg["decision_engine"]

    result_map = {r["threat_name"]: r for r in results}
    threat_names = sorted(result_map.keys())

    # ── Threat Search Selection ──
    col_sel, col_act = st.columns([4, 1])
    with col_sel:
        selected_threat = st.selectbox(
            "Target Threat Weapon",
            options=threat_names,
            index=None,
            placeholder="Type or select threat weapon...",
            key="threat_search",
        )
    with col_act:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.button("ANALYSE", width="stretch")

    # If no threat selected yet, show Decision Rules & RF Truth Table
    if selected_threat is None:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("### Decision Engine Rules & Bands")
        st.markdown(f"""
        <div class="mini-card-light">
            <table style="width:100%; color:#334155; border-collapse:collapse; font-size:0.88rem;">
                <thead>
                    <tr style="border-bottom:2px solid #e2e8f0; text-align:left; color:#0f172a;">
                        <th style="padding:10px 8px;">Condition</th>
                        <th style="padding:10px 8px;">Rule</th>
                        <th style="padding:10px 8px;">Rationale</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom:1px solid #f1f5f9;">
                        <td style="padding:12px 8px;"><span class="badge-light">gap &le; {de['SMALL_GAP']}pp &amp; Pk &ge; {de['CRITICAL_PK']}%</span></td>
                        <td style="padding:12px 8px; color:#0f172a;"><strong>Cost Wins</strong><br>Pick cheaper of top options within SMALL_GAP</td>
                        <td style="padding:12px 8px;">Effectiveness is statistically indistinguishable — no operational reason to pay more.</td>
                    </tr>
                    <tr style="border-bottom:1px solid #f1f5f9;">
                        <td style="padding:12px 8px;"><span class="badge-light">{de['SMALL_GAP']}pp &lt; gap &le; {de['LARGE_GAP']}pp &amp; Pk &ge; {de['CRITICAL_PK']}%</span></td>
                        <td style="padding:12px 8px; color:#0f172a;"><strong>Weighted Score</strong><br>Value = {de['alpha']}&middot;norm(Pk) &minus; {de['beta']}&middot;norm(Cost)</td>
                        <td style="padding:12px 8px;">Moderate gap — blend effectiveness and cost metrics.</td>
                    </tr>
                    <tr style="border-bottom:1px solid #f1f5f9;">
                        <td style="padding:12px 8px;"><span class="badge-light">gap &gt; {de['LARGE_GAP']}pp</span></td>
                        <td style="padding:12px 8px; color:#0f172a;"><strong>Pk Wins Outright</strong><br>Select highest Pk option regardless of cost</td>
                        <td style="padding:12px 8px;">Large gap in real defence terms — one option is meaningfully more likely to succeed.</td>
                    </tr>
                    <tr style="border-bottom:1px solid #f1f5f9;">
                        <td style="padding:12px 8px;"><span class="badge-light">Pk<sub>best</sub> &lt; {de['CRITICAL_PK']}%</span></td>
                        <td style="padding:12px 8px; color:#0f172a;"><strong>Pk Wins (Ignore Cost)</strong><br>Select highest Pk single option</td>
                        <td style="padding:12px 8px;">Low-confidence zone — cost savings are not justifiable when base effectiveness is unreliable.</td>
                    </tr>
                    <tr>
                        <td style="padding:12px 8px;"><span class="badge-light">Pk<sub>best</sub> &lt; {de.get('COMBO_TRIGGER_PK', 60)}%</span></td>
                        <td style="padding:12px 8px; color:#0f172a;"><strong>+ Combination Salvo</strong><br>Search 2 &amp; 3 weapon salvos for P<sub>combined</sub> &ge; {de['TARGET_PK']}%</td>
                        <td style="padding:12px 8px;">Very low confidence — single interceptor is unreliable; trigger multi-weapon salvo search.</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)



        n_total = len(results)
        n_matched = sum(1 for r in results if r.get("recommended_option"))
        st.markdown(f"""
        <div class="mini-card-light" style="text-align:center;">
            <span style="color:#0f172a; font-weight:700;">{n_total}</span> threat profiles indexed &nbsp;&bull;&nbsp;
            <span style="color:#0f172a; font-weight:700;">{n_matched}</span> threats matched to feasible counter-measures &nbsp;&bull;&nbsp;
            <span style="color:#64748b;">{n_total - n_matched}</span> space/ICBM outliers failing altitude/range constraints
        </div>
        """, unsafe_allow_html=True)
        return

    # ─────────────────────────────────────────
    #  THREAT DISPLAY SECTION
    # ─────────────────────────────────────────
    res = result_map[selected_threat]
    rec = res.get("recommended_option")
    alts = res.get("alternatives", [])

    st.markdown("---")

    # ── Threat Profile Metrics ──
    st.markdown("### Threat Profile Specs")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("TPS Rank", f"#{res['TPS_rank']}")
    m2.metric("TPS Score", f"{res['TPS']:.4f}")
    m3.metric("Max Speed", f"{res['max_speed_kmh']:,.0f} km/h")
    m4.metric("Altitude", f"{res['max_altitude_m']:,.0f} m")
    m5.metric("Detection Range", f"{res['engagement_range_km']:,.0f} km")
    m6.metric("Payload", f"{res['payload_kg']:,.1f} kg")

    rf_status_str = "True (Active RF Link Emitting)" if res["rf_link"] else "False (Autonomous Target / No RF Emission)"

    st.markdown(f"""
    <div class="mini-card-light">
        <h4>Target Weapon Identification</h4>
        <div class="val">{res['threat_name']}</div>
        <div class="sub">
            Threat Class: <strong style="color:#0f172a;">{res['threat_class']}</strong> &nbsp;|&nbsp;
            Target Live RF Link: <strong style="color:#0f172a;">{rf_status_str}</strong> &nbsp;|&nbsp;
            Feasible Counter-Measures: <strong style="color:#0f172a;">{res['num_feasible_cms']}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not rec:
        st.markdown("""
        <div class="warning-light">
            <strong>⚠ No counter-measure passed the feasibility filter for this threat.</strong><br>
            All available counter-measures failed one or more hard constraints
            (range envelope, altitude ceiling, reaction time, RF-link compatibility, or inventory).
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown("---")

    # ── Top 5 Counter-Measures Table ──
    st.markdown("### Top 5 Counter-Measures by Kill Probability (Pk)")

    if alts:
        top5 = sorted(alts, key=lambda x: x["Pk"], reverse=True)[:5]
        top5_df = pd.DataFrame([
            {
                "Rank": i + 1,
                "Counter-Measure": a["cm_name"],
                "Weapon Class": a["weapon_class"],
                "Kill Prob (Pk %)": a["Pk"],
                "Base Pk (%)": a["base_pk"],
                "Speed Ratio (%)": a.get("speed_ratio", 0),
                "Weapon Tier (%)": a.get("weapon_class_tier", 0),
                "Range Fit (%)": a["range_fit"],
                "Reaction Margin (%)": a["reaction_margin"],
                "Altitude Fit (%)": a["altitude_fit"],
                "Cost per Engagement (INR)": a["cost"],
                "Inventory": a["inventory"],
            }
            for i, a in enumerate(top5)
        ])

        st.dataframe(
            top5_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Kill Prob (Pk %)": st.column_config.ProgressColumn(
                    "Kill Prob (Pk %)", min_value=0, max_value=100, format="%.1f%%",
                ),
                "Cost per Engagement (INR)": st.column_config.NumberColumn(
                    "Cost per Engagement (INR)", format="₹%.0f",
                ),
            },
        )

        # ── Plotly Pk vs Cost Scatter (Minimalist Light) ──
        all_sorted = sorted(alts, key=lambda x: x["Pk"], reverse=True)
        chart_df = pd.DataFrame(all_sorted)

        fig = go.Figure()

        # All options
        fig.add_trace(go.Scatter(
            x=chart_df["cost"],
            y=chart_df["Pk"],
            mode="markers",
            marker=dict(size=8, color="#94a3b8", opacity=0.7,
                        line=dict(width=1, color="#cbd5e1")),
            text=chart_df["cm_name"],
            hovertemplate="<b>%{text}</b><br>Pk: %{y:.1f}%<br>Cost: ₹%{x:,.0f}<extra></extra>",
            name="Feasible Options",
        ))

        # Highlight Top 5
        top5_chart = pd.DataFrame(top5)
        fig.add_trace(go.Scatter(
            x=top5_chart["cost"],
            y=top5_chart["Pk"],
            mode="markers+text",
            marker=dict(size=13, color="#334155",
                        line=dict(width=1.5, color="#000000")),
            text=[f"#{i+1}" for i in range(len(top5_chart))],
            textposition="top center",
            textfont=dict(color="#0f172a", size=10, family="Inter"),
            hovertemplate="<b>%{customdata}</b><br>Pk: %{y:.1f}%<br>Cost: ₹%{x:,.0f}<extra></extra>",
            customdata=top5_chart["cm_name"],
            name="Top 5 Pk",
        ))

        # Highlight Selected / Recommended
        rec_names = rec["counter_measures"]
        for cm_name in rec_names:
            match = [a for a in alts if a["cm_name"] == cm_name]
            if match:
                fig.add_trace(go.Scatter(
                    x=[match[0]["cost"]],
                    y=[match[0]["Pk"]],
                    mode="markers",
                    marker=dict(size=18, color="#000000", symbol="star",
                                line=dict(width=2, color="#ffffff")),
                    name=f"★ Recommended",
                    hovertemplate=f"<b>{cm_name}</b><br>Pk: {match[0]['Pk']:.1f}%<br>Cost: ₹{match[0]['cost']:,.0f}<extra></extra>",
                ))

        # CRITICAL_PK Threshold Line
        fig.add_hline(
            y=de["CRITICAL_PK"],
            line_dash="dash", line_color="#64748b", line_width=1,
            annotation_text=f"CRITICAL_PK ({de['CRITICAL_PK']}%)",
            annotation_font_color="#475569",
            annotation_font_size=10,
        )

        fig.update_layout(
            **LIGHT_PLOTLY,
            height=360,
            title=dict(text=f"Neutralization Probability vs Engagement Cost", font=dict(color="#0f172a", size=14)),
            xaxis_title="Cost per Engagement (INR)",
            yaxis_title="Kill Probability (Pk %)",
            legend=dict(
                bgcolor="rgba(255,255,255,0.8)",
                font=dict(color="#334155", size=11),
                yanchor="top", y=0.99, xanchor="right", x=0.99,
            ),
            showlegend=True,
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    # ── Final Recommendation Section ──
    st.markdown("### Final System Recommendation")

    band = rec["decision_band_used"]
    band_label_map = {
        "cost_tiebreak": "Cost Tiebreak",
        "weighted": "Weighted Blend",
        "pk_priority": "Pk Priority",
        "low_confidence": "Low Confidence",
        "single_option": "Single Option",
    }
    b_label = band_label_map.get(band, band)

    rec_type_str = "Multi-Weapon Combination Salvo" if rec["type"] == "combination" else "Single Interceptor Engagement"

    st.markdown(f"""
    <div class="rec-box-light">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <h4 style="margin:0 !important; color:#0f172a !important; font-size:0.85rem !important; letter-spacing:0.06em; text-transform:uppercase;">
                Optimal Defence Selection
            </h4>
            <span class="badge-light">{b_label}</span>
        </div>
        <div style="font-size:1.6rem; font-weight:800; color:#0f172a;">
            {', '.join(rec['counter_measures'])}
        </div>
        <div style="font-size:0.85rem; color:#64748b; margin-top:6px;">
            Engagement Mode: <strong style="color:#0f172a;">{rec_type_str}</strong>
        </div>
        <div style="display:flex; gap:48px; margin-top:20px; border-top:1px solid #e2e8f0; padding-top:16px;">
            <div>
                <div style="color:#64748b; font-size:0.72rem; text-transform:uppercase; font-weight:700; letter-spacing:0.05em;">
                    Neutralization Probability (Pk)
                </div>
                <div style="color:#0f172a; font-size:2rem; font-weight:800; margin-top:2px;">
                    {rec['Pk']:.1f}%
                </div>
            </div>
            <div>
                <div style="color:#64748b; font-size:0.72rem; text-transform:uppercase; font-weight:700; letter-spacing:0.05em;">
                    Total Engagement Cost
                </div>
                <div style="color:#0f172a; font-size:2rem; font-weight:800; margin-top:2px;">
                    INR {rec['total_cost']:,.0f}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Justification Text
    st.markdown(f"""
    <div class="justification-light">
        <strong style="color:#0f172a;">Decision Justification:</strong><br>
        {rec['justification_text']}
    </div>
    """, unsafe_allow_html=True)

    # ── Combination Mode Details Panel ──
    combo = res.get("combination_option")
    if combo:
        target_pk = de["TARGET_PK"]
        meets = combo.get("meets_target", False)
        status_text = f"TARGET MET (&ge; {target_pk}%)" if meets else f"TARGET UNMET (&lt; {target_pk}%) — Best Achievable Salvo"

        st.markdown(f"""
        <div class="mini-card-light" style="border-color: #94a3b8;">
            <h4>Combination Mode — Salvo Analysis</h4>
            <div class="sub">
                Salvo Composition: <strong style="color:#0f172a;">{', '.join(combo['counter_measures'])}</strong><br>
                Individual Weapon Pks: <strong style="color:#0f172a;">{', '.join(f'{p:.1f}%' for p in combo['individual_pks'])}</strong><br>
                Combined Kill Probability (P<sub>combined</sub> = 1 &minus; &Pi;(1 &minus; Pk<sub>i</sub>)):
                    <strong style="color:#0f172a; font-size:1.05rem;">{combo['Pk']:.1f}%</strong><br>
                Total Salvo Cost: <strong style="color:#0f172a;">INR {combo['total_cost']:,.0f}</strong><br>
                Confidence Status: <span class="badge-light">{status_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not meets:
            st.markdown(f"""
            <div class="warning-light">
                <strong>⚠ Warning: Target neutralization probability ({target_pk}%) is not achievable with current inventory.</strong><br>
                The maximum achievable combined probability across all feasible combinations is {combo['Pk']:.1f}%.
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
