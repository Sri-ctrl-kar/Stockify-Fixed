import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# ─── Stockify Brand Colors ───────────────────
TEAL       = "#0EA5A0"
TEAL_LIGHT = "#5DD9D4"
CORAL      = "#F97316"
AMBER      = "#F59E0B"
PURPLE     = "#8B5CF6"
BG         = "#0F1117"
SURFACE    = "#1A1D2E"
TEXT       = "#E2E8F0"
MUTED      = "#64748B"
GREEN      = "#22C55E"
RED        = "#EF4444"

TEMPLATE = dict(
    layout=dict(
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(color=TEXT, family="Inter, sans-serif"),
        xaxis=dict(gridcolor="#2A2F45", linecolor="#2A2F45", zeroline=False),
        yaxis=dict(gridcolor="#2A2F45", linecolor="#2A2F45", zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
        margin=dict(l=15, r=15, t=40, b=20),
    )
)


def apply_template(fig):
    fig.update_layout(**TEMPLATE["layout"])
    return fig


def chart_top_products(product_metrics: pd.DataFrame, n: int = 10):
    """Horizontal bar chart showing top products by revenue, colored by ABC class."""
    if len(product_metrics) == 0:
        return None
        
    top = product_metrics.head(n).copy()
    top = top.sort_values("Total Revenue", ascending=True)

    colors = []
    for cls in top["ABC Class"]:
        if "A" in cls:
            colors.append(TEAL)
        elif "B" in cls:
            colors.append(AMBER)
        else:
            colors.append(MUTED)

    fig = go.Figure(go.Bar(
        y=top["Product"],
        x=top["Total Revenue"],
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color=SURFACE, width=1)
        ),
        text=top["Total Revenue"].apply(lambda v: f" ₹{v:,.0f}"),
        textposition="outside",
        textfont=dict(color=TEXT, size=10),
        hovertemplate="<b>%{y}</b><br>Revenue: ₹%{x:,.2f}<extra></extra>"
    ))
    
    fig.update_layout(**TEMPLATE["layout"])
    fig.update_layout(
        title=dict(text=f"Top {n} Products by Revenue", font=dict(size=14, color=TEXT, weight="bold")),
        xaxis=dict(title="Revenue (₹)", showgrid=True),
        yaxis=dict(title=""),
        height=max(320, n * 36)
    )
    return fig


def chart_abc_pie(product_metrics: pd.DataFrame):
    """Donut chart illustrating total revenue split by ABC classification."""
    if len(product_metrics) == 0:
        return None
        
    abc = product_metrics.groupby("ABC Class")["Total Revenue"].sum().reset_index()
    fig = px.pie(
        abc,
        names="ABC Class",
        values="Total Revenue",
        color="ABC Class",
        color_discrete_map={
            "A – High Value": TEAL,
            "B – Medium Value": AMBER,
            "C – Low Value": MUTED,
        },
        hole=0.6,
    )
    fig.update_traces(
        textfont_size=12, 
        pull=[0.02, 0, 0],
        textinfo="percent",
        hovertemplate="<b>%{label}</b><br>Revenue: ₹%{value:,.2f}<br>Share: %{percent}<extra></extra>"
    )
    fig.update_layout(**TEMPLATE["layout"])
    fig.update_layout(
        title=dict(text="ABC Revenue Contribution", font=dict(size=14, color=TEXT, weight="bold")),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig


def chart_monthly_trend(monthly_trend: pd.DataFrame):
    """Bar/Line combination chart showing monthly sales trends, MA, and volume."""
    if len(monthly_trend) == 0:
        return None
        
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Monthly revenue bars
    fig.add_trace(
        go.Bar(
            x=monthly_trend["Month"],
            y=monthly_trend["Revenue"],
            name="Monthly Revenue",
            marker_color=TEAL,
            opacity=0.8,
            hovertemplate="Revenue: ₹%{y:,.0f}<extra></extra>"
        ),
        secondary_y=False,
    )
    
    # 3-Month Moving Average trend line
    fig.add_trace(
        go.Scatter(
            x=monthly_trend["Month"],
            y=monthly_trend["Revenue 3M MA"],
            name="3-Month Trend (MA)",
            line=dict(color=CORAL, width=2.5, dash="dash"),
            mode="lines",
            hovertemplate="3M Avg: ₹%{y:,.0f}<extra></extra>"
        ),
        secondary_y=False,
    )
    
    # Total units sold line
    fig.add_trace(
        go.Scatter(
            x=monthly_trend["Month"],
            y=monthly_trend["Units"],
            name="Units Sold",
            line=dict(color=AMBER, width=2),
            mode="lines+markers",
            marker=dict(size=6, symbol="circle"),
            hovertemplate="Units Sold: %{y:,}<extra></extra>"
        ),
        secondary_y=True,
    )

    fig.update_layout(**TEMPLATE["layout"])
    fig.update_layout(
        title=dict(text="Monthly Store Performance Trend", font=dict(size=14, color=TEXT, weight="bold")),
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    
    fig.update_yaxes(title_text="Revenue (₹)", secondary_y=False, gridcolor="#2A2F45", title_font=dict(color=TEAL))
    fig.update_yaxes(title_text="Units Sold", secondary_y=True, title_font=dict(color=AMBER))
    return fig


def chart_weekly_pattern(weekly_df: pd.DataFrame):
    """Bar chart illustrating store revenue averages across days of the week."""
    if len(weekly_df) == 0:
        return None
        
    fig = go.Figure(go.Bar(
        x=weekly_df["Day"],
        y=weekly_df["Avg Revenue"],
        marker=dict(
            color=weekly_df["Avg Revenue"],
            colorscale=[[0, SURFACE], [0.6, TEAL_LIGHT], [1, TEAL]],
            showscale=False,
            line=dict(color=SURFACE, width=1)
        ),
        text=weekly_df["Avg Revenue"].apply(lambda v: f"₹{v:,.0f}"),
        textposition="outside",
        textfont=dict(color=TEXT, size=9),
        hovertemplate="Day: %{x}<br>Avg Daily Revenue: ₹%{y:,.2f}<extra></extra>"
    ))
    
    fig.update_layout(**TEMPLATE["layout"])
    fig.update_layout(
        title=dict(text="Average Revenue by Day of Week", font=dict(size=14, color=TEXT, weight="bold")),
        xaxis=dict(title=""),
        yaxis=dict(title="Avg Revenue (₹)"),
        height=320
    )
    return fig


def chart_category_breakdown(category_df: pd.DataFrame):
    """Horizontal bar chart displaying category sales share."""
    if category_df is None or len(category_df) == 0:
        return None
        
    category_df = category_df.sort_values("Revenue", ascending=True)
    
    fig = go.Figure(go.Bar(
        y=category_df["Category"],
        x=category_df["Revenue"],
        orientation="h",
        marker=dict(
            color=PURPLE,
            line=dict(color=SURFACE, width=1)
        ),
        text=category_df["Revenue"].apply(lambda v: f" ₹{v:,.0f} ({category_df.loc[category_df['Revenue']==v, 'Revenue %'].values[0]}%)"),
        textposition="outside",
        textfont=dict(color=TEXT, size=9),
        hovertemplate="<b>Category: %{y}</b><br>Revenue: ₹%{x:,.2f}<extra></extra>"
    ))
    
    fig.update_layout(**TEMPLATE["layout"])
    fig.update_layout(
        title=dict(text="Revenue Breakdown by Category", font=dict(size=14, color=TEXT, weight="bold")),
        xaxis=dict(title="Total Revenue (₹)"),
        yaxis=dict(title=""),
        height=320
    )
    return fig


def chart_forecast(prod_hist: pd.DataFrame, future_fc: pd.DataFrame, prod: str, method: str):
    """
    Renders historical sales and future demand predictions with confidence bounds.
    Matches Prophet's visual signature.
    """
    fig = go.Figure()

    # Historical scatter points
    fig.add_trace(go.Scatter(
        x=prod_hist["ds"],
        y=prod_hist["y"],
        name="Historical Daily",
        mode="markers",
        marker=dict(color="#4A5568", size=4, opacity=0.7),
        hovertemplate="Date: %{x|%d %b %Y}<br>Units Sold: %{y:.0f}<extra></extra>"
    ))

    # Upper bound confidence line
    fig.add_trace(go.Scatter(
        x=future_fc["ds"],
        y=future_fc["yhat_upper"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))

    # Lower bound confidence line, filled to the upper bound trace
    fig.add_trace(go.Scatter(
        x=future_fc["ds"],
        y=future_fc["yhat_lower"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(14, 165, 160, 0.15)",
        name="Confidence Interval (±1σ)",
        hovertemplate="Expected Range: %{customdata:.1f} - %{y:.1f} units<extra></extra>",
        customdata=future_fc["yhat_upper"]
    ))

    # Center forecasted trend line
    fig.add_trace(go.Scatter(
        x=future_fc["ds"],
        y=future_fc["yhat"],
        name=f"Forecast ({method.replace('_', ' ').title()})",
        mode="lines",
        line=dict(color=TEAL, width=3),
        hovertemplate="Date: %{x|%d %b %Y}<br>Forecasted: %{y:.1f} units<extra></extra>"
    ))

    # safety stocking reference (safety buffer visualization)
    hist_avg = prod_hist["y"].mean()
    fig.add_trace(go.Scatter(
        x=pd.concat([prod_hist["ds"], future_fc["ds"]]),
        y=[hist_avg] * (len(prod_hist) + len(future_fc)),
        name="Hist Daily Average",
        mode="lines",
        line=dict(color=MUTED, width=1.2, dash="dot"),
        hoverinfo="skip"
    ))

    fig.update_layout(**TEMPLATE["layout"])
    fig.update_layout(
        title=dict(text=f"Demand Forecast — {prod}", font=dict(size=14, color=TEXT, weight="bold")),
        xaxis=dict(title="Timeline", gridcolor="#2A2F45"),
        yaxis=dict(title="Daily Quantity Sold (Units)", gridcolor="#2A2F45"),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    return fig


def chart_slow_movers(slow_movers: pd.DataFrame):
    """Bar chart listing critically slow-moving products."""
    if len(slow_movers) == 0:
        return None
        
    top_slow = slow_movers.head(8).copy()
    top_slow = top_slow.sort_values("Total Units", ascending=False)

    colors = []
    for risk in top_slow["Risk"]:
        if "Critical" in risk:
            colors.append(RED)
        elif "Watch" in risk:
            colors.append(AMBER)
        else:
            colors.append(CORAL)

    fig = go.Figure(go.Bar(
        y=top_slow["Product"],
        x=top_slow["Total Units"],
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color=SURFACE, width=1)
        ),
        text=top_slow["Total Units"].apply(lambda v: f" {int(v)} units"),
        textposition="outside",
        textfont=dict(color=TEXT, size=9),
        hovertemplate="<b>%{y}</b><br>Units Sold: %{x:.0f}<extra></extra>"
    ))
    
    fig.update_layout(**TEMPLATE["layout"])
    fig.update_layout(
        title=dict(text="Critically Slow-Moving Products (Lowest Velocity)", font=dict(size=14, color=TEXT, weight="bold")),
        xaxis=dict(title="Total Units Sold"),
        yaxis=dict(title=""),
        height=320
    )
    return fig
