"""
Stockify v2 — AI-Powered Retail Inventory Intelligence
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import os

from core_analytics import (
    load_and_clean,
    compute_summary_kpis,
    compute_product_metrics,
    compute_monthly_trend,
    compute_weekly_pattern,
    compute_category_breakdown,
    detect_slow_movers,
    forecast_product,
    compute_reorder_alerts,
    build_ai_context,
    get_ai_insights,
)
from charts import (
    chart_top_products,
    chart_abc_pie,
    chart_monthly_trend,
    chart_weekly_pattern,
    chart_category_breakdown,
    chart_forecast,
    chart_slow_movers,
    TEAL, CORAL, AMBER, GREEN, RED, SURFACE, TEXT, MUTED,
)
from sample_data import generate_sample_data

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Stockify — Retail Inventory Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

/* Main layout overrides */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0F1117;
    color: #E2E8F0;
}

/* Glassmorphism theme panels */
div[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 80% 20%, #151a2e 0%, #0F1117 80%);
}

/* Hero header */
.hero {
    background: linear-gradient(135deg, #0EA5A0 0%, #065F5B 50%, #0F1117 100%);
    border-radius: 16px;
    padding: 2.2rem 2.8rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(14,165,160,0.15);
    border: 1px solid rgba(14,165,160,0.25);
}
.hero::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 450px;
    height: 450px;
    background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0) 70%);
    border-radius: 50%;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
    color: #ffffff;
    letter-spacing: -0.02em;
}
.hero p {
    color: rgba(255,255,255,0.85);
    font-size: 1.05rem;
    margin: 0;
    max-width: 800px;
    line-height: 1.5;
}

/* KPI Cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1.25rem;
    margin: 1.5rem 0 2rem 0;
}
.kpi-card {
    background: #1A1D2E;
    border: 1px solid rgba(14,165,160,0.18);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    position: relative;
    box-shadow: 0 6px 18px rgba(0,0,0,0.2);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}
.kpi-card:hover {
    transform: translateY(-4px);
    border-color: rgba(14,165,160,0.4);
    box-shadow: 0 12px 25px rgba(14,165,160,0.15);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    border-radius: 14px 14px 0 0;
    background: linear-gradient(90deg, #0EA5A0, #5DD9D4);
}
.kpi-card.revenue::before { background: linear-gradient(90deg, #0EA5A0, #5DD9D4); }
.kpi-card.units::before { background: linear-gradient(90deg, #F97316, #FDBA74); }
.kpi-card.products::before { background: linear-gradient(90deg, #8B5CF6, #C084FC); }
.kpi-card.daily::before { background: linear-gradient(90deg, #3B82F6, #93C5FD); }

.kpi-label {
    font-size: 0.78rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}
.kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.85rem;
    font-weight: 700;
    color: #E2E8F0;
    margin: 0.3rem 0;
}
.kpi-sub {
    font-size: 0.8rem;
    color: #64748B;
    display: flex;
    align-items: center;
    gap: 0.3rem;
}
.kpi-badge-up {
    background: rgba(34,197,94,0.15);
    color: #4ADE80;
    font-weight: 600;
    padding: 0.1rem 0.4rem;
    border-radius: 6px;
    font-size: 0.75rem;
}
.kpi-badge-down {
    background: rgba(239,68,68,0.15);
    color: #FCA5A5;
    font-weight: 600;
    padding: 0.1rem 0.4rem;
    border-radius: 6px;
    font-size: 0.75rem;
}

/* Section headers */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem;
    font-weight: 600;
    color: #F1F5F9;
    border-left: 4px solid #0EA5A0;
    padding-left: 0.85rem;
    margin: 2rem 0 1rem 0;
}

/* AI Insight box */
.ai-box {
    background: linear-gradient(135deg, #1A1D2E 0%, #10121D 100%);
    border: 1px solid rgba(14,165,160,0.4);
    border-radius: 16px;
    padding: 1.8rem 2.2rem;
    position: relative;
    box-shadow: 0 8px 24px rgba(14,165,160,0.08);
}
.ai-box::before {
    content: '✦ STOCKIFY AI ANALYST';
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    color: #5DD9D4;
    position: absolute;
    top: -12px;
    left: 24px;
    background: #0F1117;
    padding: 0 10px;
    border: 1px solid rgba(14,165,160,0.3);
    border-radius: 20px;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #0B0D14 !important;
    border-right: 1px solid rgba(14,165,160,0.15);
}

/* Tab buttons styling */
[data-testid="stTabs"] button {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    color: #94A3B8;
    transition: all 0.25s ease;
}
[data-testid="stTabs"] button:hover {
    color: #0EA5A0;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #5DD9D4 !important;
}

/* Dynamic Cards for recommended actions */
.action-card {
    background: #161926;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    border-left: 4px solid #64748B;
    transition: transform 0.2s ease;
}
.action-card:hover {
    transform: translateX(4px);
}
.action-card.critical { border-left-color: #EF4444; }
.action-card.watch { border-left-color: #F59E0B; }
.action-card.slow { border-left-color: #F97316; }

/* Custom Badge colors */
.badge {
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 600;
}
.badge-red { background: rgba(239,68,68,0.15); color: #EF4444; }
.badge-amber { background: rgba(245,158,11,0.15); color: #F59E0B; }
.badge-green { background: rgba(34,197,94,0.15); color: #22C55E; }

/* Hide default streamlit decoration */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────

for key in ["df", "col_map", "kpis", "product_metrics", "monthly_trend",
            "weekly_pattern", "category_df", "slow_movers",
            "ai_insights", "data_loaded"]:
    if key not in st.session_state:
        st.session_state[key] = None

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 2rem 0;'>
        <div style='font-family:Space Grotesk; font-size:1.9rem; font-weight:700; color:#5DD9D4; letter-spacing:-0.03em;'>📦 Stockify</div>
        <div style='font-size:0.78rem; color:#64748B; margin-top:0.25rem; text-transform:uppercase; letter-spacing:0.12em;'>AI Inventory Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Data Source")
    data_mode = st.radio(
        "",
        ["📁 Upload My Data", "🎯 Try Sample Data"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    if data_mode == "📁 Upload My Data":
        uploaded = st.file_uploader(
            "Upload Sales File",
            type=["csv", "xlsx", "xls"],
            help="CSV or Excel file with Date, Product, Quantity, and Revenue columns."
        )
        st.markdown("""
        <div style='font-size:0.75rem; color:#64748B; margin-top:0.6rem; line-height:1.4;'>
        💡 <b>Supported formats</b>: Vyapar, Tally, custom exports.<br>
        ✅ Required: <b>Date, Product, Quantity, Revenue</b><br>
        ✅ Optional: Category, Unit Price
        </div>
        """, unsafe_allow_html=True)
    else:
        uploaded = None

    st.markdown("---")
    st.markdown("### 🤖 AI Insights")
    
    # Pre-fill from environment variable or local .env file if available
    env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not env_key:
        if os.path.exists(".env"):
            try:
                with open(".env", "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            if k.strip() == "ANTHROPIC_API_KEY":
                                env_key = v.strip().strip("'\"")
                                break
            except Exception:
                pass
                
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=env_key,
        placeholder="sk-ant-...",
        help="Configure ANTHROPIC_API_KEY in your .env file or system environment to enable zero-configuration loading.",
    )
    enable_ai = bool(api_key.strip()) if api_key else False

    if enable_ai:
        st.success("✅ AI Analyst Ready")
    else:
        st.info("💡 Add API key to unlock plain-English business strategies.")

    st.markdown("---")
    st.markdown("### ⚙️ Forecast Settings")
    forecast_days = st.slider("Forecast horizon (days)", 7, 60, 30)
    top_n_products = st.slider("Products to forecast", 1, 5, 3)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#475569; text-align:center;'>
    Stockify v2.0 · Upgrade Edition<br>
    <span style='color:#0EA5A0;'>stockify.in</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class='hero'>
    <h1>📦 Stockify v2 — AI-Powered Retail Inventory Intelligence</h1>
    <p>Convert your transaction history into smart stock schedules, safety budgets, and plain-English actions instantly. Powered by Facebook Prophet statistical forecasting and Claude AI.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATA LOADING & PIPELINE
# ─────────────────────────────────────────────

load_trigger = False

if data_mode == "🎯 Try Sample Data":
    if not st.session_state.data_loaded or st.session_state.get("_mode") != "sample":
        load_trigger = True
        st.session_state["_mode"] = "sample"

elif data_mode == "📁 Upload My Data" and uploaded is not None:
    if not st.session_state.data_loaded or st.session_state.get("_file") != uploaded.name:
        load_trigger = True
        st.session_state["_file"] = uploaded.name
        st.session_state["_mode"] = "upload"


if load_trigger:
    with st.spinner("🔄 Activating analytical pipelines..."):
        try:
            if data_mode == "🎯 Try Sample Data":
                raw_df = generate_sample_data()
                col_map = {
                    "date": "Date",
                    "product": "Product",
                    "quantity": "Quantity Sold",
                    "revenue": "Revenue (₹)",
                    "category": "Category",
                    "unit_price": "Unit Price (₹)",
                }
                warnings_list = []
            else:
                raw_df, col_map, warnings_list = load_and_clean(uploaded)

            # Display ingestion warnings if any
            for w in warnings_list:
                st.warning(w)

            # Compute all analytical modules
            kpis            = compute_summary_kpis(raw_df, col_map)
            product_metrics = compute_product_metrics(raw_df, col_map)
            monthly_trend   = compute_monthly_trend(raw_df, col_map)
            weekly_pattern  = compute_weekly_pattern(raw_df, col_map)
            category_df     = compute_category_breakdown(raw_df, col_map)
            slow_movers     = detect_slow_movers(product_metrics)

            # Keep in session state
            st.session_state.df              = raw_df
            st.session_state.col_map         = col_map
            st.session_state.kpis            = kpis
            st.session_state.product_metrics = product_metrics
            st.session_state.monthly_trend   = monthly_trend
            st.session_state.weekly_pattern  = weekly_pattern
            st.session_state.category_df     = category_df
            st.session_state.slow_movers     = slow_movers
            st.session_state.data_loaded     = True
            st.session_state.ai_insights     = None  # Reset AI generation on new dataset

        except Exception as e:
            st.error(f"❌ Ingestion Pipeline Failure: {e}")
            st.stop()


# ─────────────────────────────────────────────
#  LANDING / WELCOME STATE (No Data Uploaded)
# ─────────────────────────────────────────────

if not st.session_state.data_loaded:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style='background:#1A1D2E; border-radius:12px; padding:2rem 1.5rem; text-align:center; border:1px solid rgba(14,165,160,0.1); height:100%;'>
            <div style='font-size:2.5rem; margin-bottom:0.75rem;'>📊</div>
            <div style='font-family:Space Grotesk; font-size:1.15rem; font-weight:600; margin-bottom:0.5rem; color:#5DD9D4;'>Automated Analytics</div>
            <div style='font-size:0.88rem; color:#94A3B8; line-height:1.5;'>Instant ABC category classification, transaction velocities, and MoM business indicators computed in milliseconds.</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background:#1A1D2E; border-radius:12px; padding:2rem 1.5rem; text-align:center; border:1px solid rgba(14,165,160,0.1); height:100%;'>
            <div style='font-size:2.5rem; margin-bottom:0.75rem;'>🔮</div>
            <div style='font-family:Space Grotesk; font-size:1.15rem; font-weight:600; margin-bottom:0.5rem; color:#5DD9D4;'>Demand Forecasting</div>
            <div style='font-size:0.88rem; color:#94A3B8; line-height:1.5;'>Advanced time-series forecasting mapping weekly variations, historical baselines, and seasonal shifts.</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style='background:#1A1D2E; border-radius:12px; padding:2rem 1.5rem; text-align:center; border:1px solid rgba(14,165,160,0.1); height:100%;'>
            <div style='font-size:2.5rem; margin-bottom:0.75rem;'>🧠</div>
            <div style='font-family:Space Grotesk; font-size:1.15rem; font-weight:600; margin-bottom:0.5rem; color:#5DD9D4;'>AI-Generated Action Plans</div>
            <div style='font-size:0.88rem; color:#94A3B8; line-height:1.5;'>Paste your Anthropic key to unlock a complete business strategy, top performer assessments, and weekly action points.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("👈 **Get started**: Upload a transaction history file (CSV/Excel) or click 'Try Sample Data' in the sidebar to explore.")
    st.stop()


# ─────────────────────────────────────────────
#  MAIN DASHBOARD (Data Active)
# ─────────────────────────────────────────────

kpis            = st.session_state.kpis
product_metrics = st.session_state.product_metrics
monthly_trend   = st.session_state.monthly_trend
weekly_pattern  = st.session_state.weekly_pattern
category_df     = st.session_state.category_df
slow_movers     = st.session_state.slow_movers
df              = st.session_state.df
col_map         = st.session_state.col_map

# Sample data indicator badge
if st.session_state.get("_mode") == "sample":
    st.markdown("""
    <div style='background:rgba(14,165,160,0.1); border:1px solid rgba(14,165,160,0.3); border-radius:10px; padding:0.8rem 1.2rem; margin-bottom:1.5rem;'>
        <span style='color:#5DD9D4; font-weight:600;'>📋 Sample Data Active</span> — Rendering 17 months of simulated Indian retail records. Feel free to upload your store files to analyze real transactions!
    </div>
    """, unsafe_allow_html=True)

# ── KPI Cards ──
st.markdown("<div class='section-header'>📊 Overview Indicators</div>", unsafe_allow_html=True)

# Generate HTML for premium KPI Cards
growth_badge = ""
if kpis["mom_growth"] is not None:
    if kpis["mom_growth"] >= 0:
        growth_badge = f"<span class='kpi-badge-up'>↑ {kpis['mom_growth']:+.1f}% MoM</span>"
    else:
        growth_badge = f"<span class='kpi-badge-down'>↓ {kpis['mom_growth']:+.1f}% MoM</span>"

st.markdown(f"""
<div class='kpi-grid'>
    <div class='kpi-card revenue'>
        <div class='kpi-label'>Total Revenue</div>
        <div class='kpi-value'>₹{kpis['total_revenue']:,.0f}</div>
        <div class='kpi-sub'>{growth_badge} from last month</div>
    </div>
    <div class='kpi-card units'>
        <div class='kpi-label'>Units Distributed</div>
        <div class='kpi-value'>{kpis['total_units']:,}</div>
        <div class='kpi-sub'>Across daily order transactions</div>
    </div>
    <div class='kpi-card products'>
        <div class='kpi-label'>Unique Catalog Items</div>
        <div class='kpi-value'>{kpis['total_products']}</div>
        <div class='kpi-sub'>Classified under ABC levels</div>
    </div>
    <div class='kpi-card daily'>
        <div class='kpi-label'>Avg Daily Sales</div>
        <div class='kpi-value'>₹{kpis['avg_daily_rev']:,.0f}</div>
        <div class='kpi-sub'>Basket value: ₹{kpis['avg_order_value']:,.1f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style='font-size:0.8rem; color:#64748B; margin-top:-1.2rem; margin-bottom:1.5rem; text-align:right;'>
📅 Transaction Timeline: <b>{kpis['date_min'].strftime('%d %b %Y')}</b> → <b>{kpis['date_max'].strftime('%d %b %Y')}</b> ({kpis['date_range_days']} active days)
</div>
""", unsafe_allow_html=True)


# ── AI INSIGHTS SECTION ──
st.markdown("<div class='section-header'>🧠 Neural AI Business Auditor</div>", unsafe_allow_html=True)

if enable_ai:
    if st.session_state.ai_insights is None:
        if st.button("✨ Execute AI Audit", type="primary"):
            with st.spinner("🤖 Claude is analyzing your stock logs and formulating strategies..."):
                try:
                    context = build_ai_context(kpis, product_metrics, monthly_trend,
                                               slow_movers, category_df)
                    insights = get_ai_insights(context, api_key.strip())
                    st.session_state.ai_insights = insights
                    st.rerun()
                except Exception as e:
                    st.error(f"API Error: {e}")
    else:
        st.markdown("<div class='ai-box'>", unsafe_allow_html=True)
        st.markdown(st.session_state.ai_insights)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Audit"):
            st.session_state.ai_insights = None
            st.rerun()
else:
    st.markdown("""
    <div style='background:#1A1D2E; border:1px dashed rgba(14,165,160,0.3); border-radius:12px;
         padding:1.4rem 1.6rem; color:#94A3B8; font-size:0.88rem; line-height:1.5; display:flex; align-items:center; gap:1rem;'>
        <div style='font-size:1.8rem;'>🔑</div>
        <div>
            <b>AI Business Insights Locked</b> — Paste your Anthropic API key in the sidebar to generate
            professional action plans, customer demand explanations, and inventory optimizations generated by Claude 3.5 Sonnet.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── DASHBOARD TABS ──
st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📦 Product Valuation", "📈 Sales Timelines", "🔮 Demand Forecasting", "⚠️ Actionable Alerts", "🗂️ Dataset Logs"
])


# ─────────────────────────────────────────────
#  TAB 1: PRODUCT VALUATION
# ─────────────────────────────────────────────

with tab1:
    col_a, col_b = st.columns([2, 1])

    with col_a:
        n_show = st.slider("Filter Top Products", 5, min(25, len(product_metrics)), 10, key="topn_slider")
        fig_top = chart_top_products(product_metrics, n=n_show)
        if fig_top:
            st.plotly_chart(fig_top, use_container_width=True)

    with col_b:
        fig_abc = chart_abc_pie(product_metrics)
        if fig_abc:
            st.plotly_chart(fig_abc, use_container_width=True)

        st.markdown("<div class='section-header' style='font-size:0.95rem; margin-top:1.5rem;'>ABC Action Standard</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:0.84rem; color:#94A3B8; line-height:1.75;'>
        🟢 <b style='color:#5DD9D4;'>Class A (High Value)</b>: Top 70% of store revenue. Maintain high safety stocks to prevent stock-outs.<br>
        🟡 <b style='color:#F59E0B;'>Class B (Medium Value)</b>: Next 20% of revenue. Place orders periodically based on safety limits.<br>
        ⚫ <b style='color:#64748B;'>Class C (Low Value)</b>: Bottom 10% of revenue. Keep minimal stock, reduce order sizes to free up cash.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Stock Catalog & Value Ranks</div>", unsafe_allow_html=True)

    display_cols = ["Product", "Total Units", "Total Revenue", "Revenue Share %",
                    "Avg Daily Units", "ABC Class"]
    if "Category" in product_metrics.columns:
        display_cols.insert(1, "Category")

    st.dataframe(
        product_metrics[display_cols].style.format({
            "Total Revenue": "₹{:,.0f}",
            "Revenue Share %": "{:.1f}%",
            "Avg Daily Units": "{:.1f}",
        }).background_gradient(subset=["Total Revenue"], cmap="YlGnBu"),
        use_container_width=True,
        height=380,
    )


# ─────────────────────────────────────────────
#  TAB 2: SALES TIMELINES
# ─────────────────────────────────────────────

with tab2:
    fig_monthly = chart_monthly_trend(monthly_trend)
    if fig_monthly:
        st.plotly_chart(fig_monthly, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_weekly = chart_weekly_pattern(weekly_pattern)
        if fig_weekly:
            st.plotly_chart(fig_weekly, use_container_width=True)
    with col_b:
        if category_df is not None:
            fig_cat = chart_category_breakdown(category_df)
            if fig_cat:
                st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("💡 Category tracking locked. Include a 'Category' column in your data file to render this visual.")

    # Multi-product comparative timelines
    st.markdown("<div class='section-header'>Multi-Product Timeline Comparator</div>", unsafe_allow_html=True)
    all_products = product_metrics["Product"].tolist()
    selected_products = st.multiselect(
        "Choose Products to compare daily sales",
        all_products,
        default=all_products[:4] if len(all_products) >= 4 else all_products,
        max_selections=6,
    )

    if selected_products:
        date_col = col_map["date"]
        qty_col  = col_map["quantity"]
        prod_col = col_map["product"]

        filtered = df[df[prod_col].isin(selected_products)].copy()
        filtered["_month"] = filtered[date_col].dt.to_period("M")
        pivot = filtered.groupby(["_month", prod_col])[qty_col].sum().reset_index()
        pivot["Month"] = pivot["_month"].dt.to_timestamp()

        import plotly.express as px
        fig_line = px.line(
            pivot, x="Month", y=qty_col, color=prod_col,
            title="Monthly Sales Comparison (Units Sold)",
            labels={qty_col: "Units Distributed", prod_col: "Product"},
            template="plotly_dark",
        )
        fig_line.update_layout(
            plot_bgcolor="#1A1D2E", paper_bgcolor="#1A1D2E",
            font=dict(color="#E2E8F0", family="Inter, sans-serif"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="#2A2F45"),
            yaxis=dict(gridcolor="#2A2F45"),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_line, use_container_width=True)


# ─────────────────────────────────────────────
#  TAB 3: DEMAND FORECASTING
# ─────────────────────────────────────────────

with tab3:
    st.markdown("""
    <div style='background:#1A1D2E; border-radius:10px; padding:1.2rem 1.5rem; margin-bottom:1.5rem;
         border-left:4px solid #0EA5A0; font-size:0.88rem; color:#94A3B8; line-height:1.55;'>
        🔮 <b>Intelligent Statistical Forecasting Engine</b><br>
        Computes 30-day product demand models. Uses <b>Facebook Prophet</b> for full trend fitting when 14+ daily data points exist. If Prophet is not installed, it automatically transitions to a <b>Weekly-Weighted Moving Average</b> model that maps weekday variations and growth curves.
    </div>
    """, unsafe_allow_html=True)

    all_products  = product_metrics["Product"].tolist()
    default_prods = all_products[:min(top_n_products, len(all_products))]
    
    forecast_products = st.multiselect(
        "Choose Products for Demand Forecasting",
        all_products,
        default=default_prods,
        max_selections=5,
    )

    if forecast_products:
        if st.button("🔮 Calculate Horizon Forecasts", type="primary"):
            date_col = col_map["date"]
            qty_col  = col_map["quantity"]
            prod_col = col_map["product"]

            forecast_results = {}
            progress = st.progress(0)

            for i, prod in enumerate(forecast_products):
                with st.spinner(f"Modeling demand curves for {prod}..."):
                    fc_df, method = forecast_product(df, col_map, prod, periods=forecast_days)
                    forecast_results[prod] = (fc_df, method)
                    progress.progress((i + 1) / len(forecast_products))

            progress.empty()
            st.session_state["forecast_results"] = forecast_results

        if "forecast_results" in st.session_state and st.session_state["forecast_results"]:
            forecast_results = st.session_state["forecast_results"]

            for prod, (fc_df, method) in forecast_results.items():
                if fc_df is None:
                    st.warning(f"⚠️ {prod}: Insufficient sales history (needs at least 14 days of data to compute).")
                    continue

                # Historical data aggregation
                prod_hist = df[df[col_map["product"]] == prod].groupby(col_map["date"])[col_map["quantity"]].sum().reset_index()
                prod_hist.columns = ["ds", "y"]

                # Future predictions only
                future_fc = fc_df[fc_df["ds"] > prod_hist["ds"].max()]

                col_l, col_r = st.columns([3.2, 1])
                with col_l:
                    fig_fc = chart_forecast(prod_hist, future_fc, prod, method)
                    if fig_fc:
                        st.plotly_chart(fig_fc, use_container_width=True)
                with col_r:
                    avg_forecast = future_fc["yhat"].mean()
                    total_forecast = future_fc["yhat"].sum()
                    hist_avg = prod_hist["y"].tail(30).mean()
                    change = ((avg_forecast - hist_avg) / max(hist_avg, 0.01)) * 100

                    st.markdown(f"""
                    <div style='background:#1A1D2E; border-radius:12px; padding:1.4rem; border:1px solid rgba(14,165,160,0.15); margin-top:2.5rem;'>
                        <div style='font-family:Space Grotesk; font-size:1.15rem; font-weight:700; color:#5DD9D4; margin-bottom:0.75rem; border-bottom:1px solid rgba(14,165,160,0.15); padding-bottom:0.5rem;'>{prod}</div>
                        <div style='font-size:0.78rem; color:#64748B; text-transform:uppercase;'>Compute Engine</div>
                        <div style='font-size:0.9rem; font-weight:600; margin-bottom:0.75rem; color:#E2E8F0;'>{method.replace("_", " ").title()}</div>
                        <div style='font-size:0.78rem; color:#64748B; text-transform:uppercase;'>Daily Sales Speed</div>
                        <div style='font-family:Space Grotesk; font-size:1.4rem; font-weight:700; color:#0EA5A0; margin-bottom:0.75rem;'>{avg_forecast:.1f} units</div>
                        <div style='font-size:0.78rem; color:#64748B; text-transform:uppercase;'>Est. {forecast_days}-Day Stock Needs</div>
                        <div style='font-size:1.15rem; font-weight:600; margin-bottom:0.75rem; color:#E2E8F0;'>{int(total_forecast)} units</div>
                        <div style='font-size:0.78rem; color:#64748B; text-transform:uppercase;'>Trend shift vs 30d avg</div>
                        <div style='font-size:1rem; font-weight:700; color:{"#4ADE80" if change >= 0 else "#FCA5A5"};'>{change:+.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("💡 Select products above and click 'Calculate Horizon Forecasts' to view predictions.")


# ─────────────────────────────────────────────
#  TAB 4: ACTIONABLE ALERTS
# ─────────────────────────────────────────────

with tab4:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<div class='section-header'>⚠️ Low-Velocity Stock (Slow Movers)</div>",
                    unsafe_allow_html=True)

        if len(slow_movers) > 0:
            fig_slow = chart_slow_movers(slow_movers)
            if fig_slow:
                st.plotly_chart(fig_slow, use_container_width=True)

            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
            st.markdown("<b>Auditor Recommendations:</b>", unsafe_allow_html=True)
            for _, row in slow_movers.head(5).iterrows():
                risk_class = "slow"
                badge_class = "badge-amber"
                if "Critical" in row["Risk"]:
                    risk_class = "critical"
                    badge_class = "badge-red"
                elif "Watch" in row["Risk"]:
                    risk_class = "watch"
                    
                st.markdown(f"""
                <div class='action-card {risk_class}'>
                    <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem;'>
                        <b style='font-size:0.95rem; color:#F1F5F9;'>{row['Product']}</b>
                        <span class='badge {badge_class}'>{row['Risk'].split()[-1]} Risk</span>
                    </div>
                    <span style='font-size:0.8rem; color:#94A3B8;'>
                    Only {int(row['Total Units'])} units sold over the historical period. Consider promotional bundles, shelf redesigns, or temporary discount events to recover capital.
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("🎉 Outstanding inventory speed! No slow-moving items detected.")

    with col_b:
        st.markdown("<div class='section-header'>📋 Smart Reorder Schedule</div>",
                    unsafe_allow_html=True)

        # Grab forecasted daily demands if available
        fc_map = None
        if "forecast_results" in st.session_state and st.session_state["forecast_results"]:
            fc_map = {}
            for prod, (fc_df, _) in st.session_state["forecast_results"].items():
                if fc_df is not None:
                    fc_map[prod] = fc_df["yhat"].mean()

        reorder_df = compute_reorder_alerts(product_metrics, fc_map)

        for _, row in reorder_df.head(12).iterrows():
            status_color = "#64748B"
            badge_text = row["Status"]
            badge_class = "badge-green"
            
            if "Now" in row["Status"]:
                status_color = "#EF4444"
                badge_class = "badge-red"
            elif "Soon" in row["Status"]:
                status_color = "#F59E0B"
                badge_class = "badge-amber"
                
            st.markdown(f"""
            <div style='background:#1A1D2E; border-radius:10px; padding:0.8rem 1.2rem;
                 margin-bottom:0.5rem; display:flex; justify-content:space-between; align-items:center; border:1px solid rgba(14,165,160,0.1);'>
                <div>
                    <b style='font-size:0.95rem; color:#F1F5F9;'>{row['Product']}</b>
                    <span style='font-size:0.75rem; color:#64748B; margin-left:0.5rem; text-transform:uppercase;'>{row['ABC Class'].split()[0]} Category</span><br>
                    <span style='font-size:0.8rem; color:#94A3B8;'>
                        Reorder Point: <b>{row['Reorder Point (units)']} units</b> (7d) · Suggest Order: <b>{row['Suggested Order (units)']} units</b> (14d)
                    </span>
                </div>
                <span class='badge {badge_class}' style='font-size:0.78rem;'>{badge_text}</span>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  TAB 5: DATASET LOGS
# ─────────────────────────────────────────────

with tab5:
    st.markdown(f"📊 **Store Sales Logs** · {len(df):,} total records loaded.")

    col_filter, col_date = st.columns(2)
    with col_filter:
        prod_filter = st.multiselect(
            "Filter catalog items",
            df[col_map["product"]].unique().tolist(),
            default=[],
            key="raw_filter_multiselect"
        )
    with col_date:
        if kpis:
            date_range = st.date_input(
                "Filter date span",
                value=(kpis["date_min"].date(), kpis["date_max"].date()),
                key="raw_dates_input"
            )

    filtered_df = df.copy()
    if prod_filter:
        filtered_df = filtered_df[filtered_df[col_map["product"]].isin(prod_filter)]
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df[col_map["date"]].dt.date >= date_range[0]) &
            (filtered_df[col_map["date"]].dt.date <= date_range[1])
        ]

    st.dataframe(filtered_df, use_container_width=True, height=420)

    # Export capabilities
    csv_data = filtered_df.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download Filtered Data as CSV",
        csv_data,
        "stockify_export.csv",
        "text/csv",
        type="secondary",
    )
