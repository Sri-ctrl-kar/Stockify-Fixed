import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
import urllib.request

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  DATA LOADING & CLEANING
# ─────────────────────────────────────────────

REQUIRED_COLS_MAP = {
    "date":     ["date", "Date", "DATE", "Order Date", "Sale Date", "Transaction Date", "बिक्री तिथि", "तिथि"],
    "product":  ["product", "Product", "PRODUCT", "Item", "Item Name", "Product Name", "उत्पाद", "वस्तु"],
    "quantity": ["quantity", "Quantity", "QTY", "qty", "Qty Sold", "Quantity Sold", "Units Sold", "मात्रा"],
    "revenue":  ["revenue", "Revenue", "REVENUE", "Sales", "Total", "Amount", "Revenue (₹)", "राजस्व", "बिक्री"],
}

OPTIONAL_COLS_MAP = {
    "category":   ["category", "Category", "CATEGORY", "Type", "श्रेणी", "वर्ग"],
    "unit_price": ["unit_price", "Unit Price", "Price", "Unit Price (₹)", "MRP", "मूल्य", "दर"],
}


def detect_column(df: pd.DataFrame, aliases: list) -> str | None:
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


def load_and_clean(uploaded_file) -> tuple[pd.DataFrame, dict, list[str]]:
    """
    Load file, map columns, clean data.
    Returns (cleaned_df, column_map, warnings_list)
    """
    warnings_list = []
    ext = uploaded_file.name.split(".")[-1].lower()

    try:
        if ext == "csv":
            df = pd.read_csv(uploaded_file)
        elif ext in ("xls", "xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError(f"Unsupported file type: .{ext}. Please upload CSV or Excel.")
    except Exception as e:
        raise ValueError(f"Could not read uploaded file: {str(e)}")

    df.columns = df.columns.str.strip()

    col_map = {}
    for key, aliases in REQUIRED_COLS_MAP.items():
        found = detect_column(df, aliases)
        if found is None:
            raise ValueError(
                f"Could not find '{key}' column. Expected one of: {aliases[:4]}. "
                f"Your columns: {list(df.columns)}"
            )
        col_map[key] = found

    for key, aliases in OPTIONAL_COLS_MAP.items():
        found = detect_column(df, aliases)
        col_map[key] = found  # may be None

    # Parse date
    df[col_map["date"]] = pd.to_datetime(df[col_map["date"]], errors="coerce")
    bad_dates = df[col_map["date"]].isna().sum()
    if bad_dates > 0:
        warnings_list.append(f"⚠️ {bad_dates} rows had unparseable dates and were dropped.")
    df = df.dropna(subset=[col_map["date"]])

    # Clean numeric columns
    for key in ("quantity", "revenue"):
        col = col_map[key]
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(r"[₹,\s]", "", regex=True)
        df[col] = pd.to_numeric(df[col], errors="coerce")
        neg = (df[col] < 0).sum()
        if neg > 0:
            warnings_list.append(f"⚠️ {neg} rows with negative {key} were dropped.")
        df = df.dropna(subset=[col])
        df = df[df[col] >= 0]

    # Clean optional unit price
    if col_map.get("unit_price"):
        col = col_map["unit_price"]
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(r"[₹,\s]", "", regex=True)
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(0.0)
    else:
        # If no unit price, calculate it from revenue and quantity
        df["Unit Price (Calculated)"] = (df[col_map["revenue"]] / df[col_map["quantity"]].replace(0, 1)).round(2)
        col_map["unit_price"] = "Unit Price (Calculated)"

    return df.reset_index(drop=True), col_map, warnings_list


# ─────────────────────────────────────────────
#  CORE ANALYTICS
# ─────────────────────────────────────────────

def compute_summary_kpis(df: pd.DataFrame, col_map: dict) -> dict:
    date_col = col_map["date"]
    qty_col  = col_map["quantity"]
    rev_col  = col_map["revenue"]
    prod_col = col_map["product"]

    # Ensure date column is datetime
    if df[date_col].dtype == object:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    total_revenue   = df[rev_col].sum()
    total_units     = df[qty_col].sum()
    total_products  = df[prod_col].nunique()
    date_range_days = (df[date_col].max() - df[date_col].min()).days + 1
    avg_daily_rev   = total_revenue / max(date_range_days, 1)
    avg_order_value = total_revenue / max(len(df), 1)

    # MoM growth (last 2 complete months)
    df["_month"] = df[date_col].dt.to_period("M")
    monthly = df.groupby("_month")[rev_col].sum().sort_index()
    mom_growth = None
    if len(monthly) >= 2:
        prev, curr = monthly.iloc[-2], monthly.iloc[-1]
        if prev > 0:
            mom_growth = ((curr - prev) / prev) * 100

    return {
        "total_revenue":   total_revenue,
        "total_units":     int(total_units),
        "total_products":  total_products,
        "date_range_days": date_range_days,
        "avg_daily_rev":   avg_daily_rev,
        "avg_order_value": avg_order_value,
        "mom_growth":      mom_growth,
        "date_min":        df[date_col].min(),
        "date_max":        df[date_col].max(),
    }


def compute_product_metrics(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    prod_col = col_map["product"]
    qty_col  = col_map["quantity"]
    rev_col  = col_map["revenue"]
    cat_col  = col_map.get("category")

    agg = {qty_col: "sum", rev_col: "sum"}
    if cat_col:
        agg[cat_col] = "first"

    pm = df.groupby(prod_col).agg(agg).reset_index()
    pm.columns = ["Product", "Total Units", "Total Revenue"] + (["Category"] if cat_col else [])
    
    total_days = max((df[col_map["date"]].max() - df[col_map["date"]].min()).days, 1)
    pm["Avg Daily Units"] = pm["Total Units"] / total_days
    pm["Revenue Share %"] = (pm["Total Revenue"] / pm["Total Revenue"].sum() * 100).round(1)

    # ABC Classification
    pm = pm.sort_values("Total Revenue", ascending=False)
    pm["Cumulative %"] = pm["Total Revenue"].cumsum() / pm["Total Revenue"].sum() * 100
    pm["ABC Class"] = pm["Cumulative %"].apply(
        lambda x: "A – High Value" if x <= 70 else ("B – Medium Value" if x <= 90 else "C – Low Value")
    )

    return pm.reset_index(drop=True)


def compute_monthly_trend(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    date_col = col_map["date"]
    qty_col  = col_map["quantity"]
    rev_col  = col_map["revenue"]

    df["_month"] = df[date_col].dt.to_period("M")
    monthly = df.groupby("_month").agg(
        Units=(qty_col, "sum"),
        Revenue=(rev_col, "sum")
    ).reset_index()
    monthly["Month"] = monthly["_month"].dt.to_timestamp()
    monthly = monthly.sort_values("Month")
    monthly["Revenue 3M MA"] = monthly["Revenue"].rolling(3, min_periods=1).mean()
    return monthly


def compute_weekly_pattern(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    date_col = col_map["date"]
    qty_col  = col_map["quantity"]
    rev_col  = col_map["revenue"]

    df["_dow"] = df[date_col].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekly = df.groupby("_dow").agg(
        Avg_Units=(qty_col, "mean"),
        Avg_Revenue=(rev_col, "mean")
    ).reindex(order).reset_index()
    weekly.columns = ["Day", "Avg Units", "Avg Revenue"]
    return weekly


def compute_category_breakdown(df: pd.DataFrame, col_map: dict) -> pd.DataFrame | None:
    cat_col = col_map.get("category")
    if not cat_col:
        return None
    rev_col = col_map["revenue"]
    qty_col = col_map["quantity"]

    cat = df.groupby(cat_col).agg(
        Revenue=(rev_col, "sum"),
        Units=(qty_col, "sum"),
        Products=(col_map["product"], "nunique")
    ).reset_index()
    cat.columns = ["Category", "Revenue", "Units", "Products"]
    cat["Revenue %"] = (cat["Revenue"] / cat["Revenue"].sum() * 100).round(1)
    return cat.sort_values("Revenue", ascending=False).reset_index(drop=True)


def detect_slow_movers(product_metrics: pd.DataFrame, threshold_pct: float = 20.0) -> pd.DataFrame:
    """Bottom N% by units are slow movers."""
    if len(product_metrics) == 0:
        return pd.DataFrame()
        
    cutoff = product_metrics["Total Units"].quantile(threshold_pct / 100)
    slow = product_metrics[product_metrics["Total Units"] <= cutoff].copy()
    
    p05 = product_metrics["Total Units"].quantile(0.05)
    p10 = product_metrics["Total Units"].quantile(0.10)
    
    slow["Risk"] = slow["Total Units"].apply(
        lambda x: "🔴 Critical" if x <= p05
        else ("🟡 Watch" if x <= p10 else "🟠 Slow")
    )
    return slow.sort_values("Total Units")


# ─────────────────────────────────────────────
#  DEMAND FORECASTING (Prophet + Fallback)
# ─────────────────────────────────────────────

def forecast_product(df: pd.DataFrame, col_map: dict, product_name: str,
                     periods: int = 30) -> tuple[pd.DataFrame | None, str]:
    """
    Forecast next `periods` days for a given product.
    Returns (forecast_df, method_used)
    """
    date_col = col_map["date"]
    qty_col  = col_map["quantity"]
    prod_col = col_map["product"]

    prod_df = df[df[prod_col] == product_name].copy()
    daily = prod_df.groupby(date_col)[qty_col].sum().reset_index()
    daily.columns = ["ds", "y"]
    daily = daily.sort_values("ds")

    if len(daily) < 14:
        return None, "insufficient_data"

    # Fill missing dates in historical timeline with 0
    full_range = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
    daily = daily.set_index("ds").reindex(full_range, fill_value=0.0).reset_index()
    daily.columns = ["ds", "y"]

    try:
        from prophet import Prophet
        
        # Suppress logging info from Prophet
        import logging
        logging.getLogger('prophet').setLevel(logging.ERROR)
        
        m = Prophet(
            yearly_seasonality=len(daily) > 180,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
        )
        m.fit(daily)
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)
        
        # Clamp values to non-negative bounds
        forecast["yhat"] = forecast["yhat"].clip(lower=0.0)
        forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0.0)
        forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0.0)
        
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], "prophet"

    except Exception:
        # Fallback: Advanced Weekly-Weighted Moving Average with linear trend
        last_date = daily["ds"].max()
        
        # Calculate overall 30-day baseline velocity
        recent_30 = daily.tail(30)
        base_average = recent_30["y"].mean()
        
        # Calculate weekday distribution index
        daily["_dow"] = daily["ds"].dt.dayofweek
        dow_agg = daily.groupby("_dow")["y"].mean()
        overall_mean = dow_agg.mean() if dow_agg.mean() > 0 else 1.0
        dow_weights = (dow_agg / overall_mean).to_dict()
        
        # Simple trend slope based on recent vs overall
        recent_14 = daily.tail(14)["y"].mean()
        older_14 = daily.tail(28).head(14)["y"].mean()
        diff = recent_14 - older_14
        daily_growth = (diff / 14) * 0.1  # dampened growth factor
        daily_growth = np.clip(daily_growth, -0.05, 0.05)  # clamp to protect extrapolation

        future_dates = pd.date_range(last_date + timedelta(days=1), periods=periods)
        yhat_list = []
        lower_list = []
        upper_list = []
        
        for idx, date in enumerate(future_dates, 1):
            dow = date.dayofweek
            weight = dow_weights.get(dow, 1.0)
            
            # Project base with dampened trend and noise
            projected_base = base_average + (idx * daily_growth)
            projected_base = max(0.0, projected_base)
            
            # Apply weekday weight
            yhat = projected_base * weight
            
            # Apply standard deviation variance for confidence interval
            std_dev = recent_30["y"].std()
            if pd.isna(std_dev) or std_dev == 0:
                std_dev = base_average * 0.35
                
            variance = std_dev * (1.0 + (idx * 0.05)) # confidence band widens over time
            
            yhat_list.append(float(yhat))
            lower_list.append(float(max(0.0, yhat - variance)))
            upper_list.append(float(yhat + variance))
            
        fallback = pd.DataFrame({
            "ds": future_dates,
            "yhat": yhat_list,
            "yhat_lower": lower_list,
            "yhat_upper": upper_list,
        })
        return fallback, "moving_average"


def compute_reorder_alerts(product_metrics: pd.DataFrame,
                           forecast_map: dict | None = None) -> pd.DataFrame:
    """
    Generate reorder recommendations based on velocity and optional forecast.
    forecast_map: {product_name: avg_daily_units_next_30d}
    """
    alerts = product_metrics[["Product", "Total Units", "Avg Daily Units", "ABC Class"]].copy()

    # Determine forecasted rate (falls back to historical daily if unavailable)
    if forecast_map:
        alerts["Forecasted Avg Daily"] = alerts["Product"].map(forecast_map).fillna(alerts["Avg Daily Units"])
    else:
        alerts["Forecasted Avg Daily"] = alerts["Avg Daily Units"]

    # Reorder Point = 7-day safety stock
    alerts["Reorder Point (units)"] = (alerts["Forecasted Avg Daily"] * 7).round(0).astype(int)
    
    # Suggested Order = 14-day supply
    alerts["Suggested Order (units)"] = (alerts["Forecasted Avg Daily"] * 14).round(0).astype(int)

    # Status heuristic: High Value (A-class) items are monitored closely.
    alerts["Status"] = alerts.apply(
        lambda r: "🔴 Order Now" if r["ABC Class"] == "A – High Value" and r["Avg Daily Units"] > 4
        else ("🟡 Order Soon" if r["Avg Daily Units"] > 1.2 else "🟢 OK"),
        axis=1
    )
    return alerts.sort_values("Forecasted Avg Daily", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
#  AI INSIGHTS GENERATOR (Zero Dependency URLLIB)
# ─────────────────────────────────────────────

def build_ai_context(kpis: dict, product_metrics: pd.DataFrame,
                     monthly_trend: pd.DataFrame, slow_movers: pd.DataFrame,
                     category_df: pd.DataFrame | None) -> str:
    """Build a concise data summary to send to Claude."""

    top5 = product_metrics.head(5)[["Product", "Total Units", "Total Revenue", "ABC Class"]].to_string(index=False)
    bottom5 = product_metrics.tail(5)[["Product", "Total Units", "Total Revenue"]].to_string(index=False)

    trend_summary = ""
    if len(monthly_trend) >= 3:
        recent = monthly_trend.tail(3)
        trend_summary = recent[["Month", "Revenue"]].to_string(index=False)

    slow_summary = ""
    if len(slow_movers) > 0:
        slow_summary = slow_movers[["Product", "Total Units", "Risk"]].head(5).to_string(index=False)

    cat_summary = ""
    if category_df is not None:
        cat_summary = category_df[["Category", "Revenue %"]].to_string(index=False)

    mom_str = f"{kpis['mom_growth']:.1f}%" if kpis["mom_growth"] is not None else "N/A"

    context = f"""
RETAIL STORE SALES DATA SUMMARY
================================
Period: {kpis['date_min'].strftime('%d %b %Y')} to {kpis['date_max'].strftime('%d %b %Y')} ({kpis['date_range_days']} days)
Total Revenue: ₹{kpis['total_revenue']:,.0f}
Total Units Sold: {kpis['total_units']:,}
Unique Products: {kpis['total_products']}
Avg Daily Revenue: ₹{kpis['avg_daily_rev']:,.0f}
Month-over-Month Revenue Growth: {mom_str}

TOP 5 PRODUCTS (by revenue):
{top5}

BOTTOM 5 PRODUCTS (by revenue):
{bottom5}

SLOW-MOVING ITEMS (bottom 20% by units):
{slow_summary if slow_summary else 'None identified'}

RECENT MONTHLY REVENUE TREND (last 3 months):
{trend_summary if trend_summary else 'Insufficient data'}

CATEGORY BREAKDOWN:
{cat_summary if cat_summary else 'Category data not available'}
"""
    return context.strip()


AI_SYSTEM_PROMPT = """You are Stockify AI — an expert retail inventory analyst for small Indian retailers.
Your job is to analyse sales data and give CLEAR, ACTIONABLE insights in plain English.

Rules:
1. Speak like a friendly business advisor, not a data scientist
2. Use ₹ for Indian Rupees
3. Be specific — mention actual product names, numbers, percentages from the data
4. Structure your response with these EXACT sections using markdown:

## 🏆 Overall Performance
2-3 sentences on the business health.

## 📈 Top Performers
Which products are driving revenue. What the retailer should do (increase stock, feature prominently).

## ⚠️ Products Needing Attention
Slow movers or declining items. Concrete action for each (discount, stop stocking, bundle).

## 🔮 Key Trend Spotted
1-2 seasonal or demand patterns visible in the data. What to prepare for.

## ✅ This Week's Action Plan
3 specific, numbered actions the retailer should take THIS WEEK based on the data.

Keep the total response under 400 words. Be direct and confident. No hedging."""


def get_ai_insights(context: str, api_key: str) -> str:
    """Call Claude API for plain-English insights using built-in urllib to remain zero-dependency."""
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    data = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1000,
        "system": AI_SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": f"Analyse this retail data and give insights:\n\n{context}"}
        ]
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            return res_json["content"][0]["text"]
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8")
        try:
            err_json = json.loads(err_msg)
            msg = err_json.get("error", {}).get("message", "HTTP error")
        except Exception:
            msg = err_msg
        raise Exception(f"Claude API failed: {msg}")
    except Exception as e:
        raise Exception(f"AI API request failed: {e}")
