"""
build_report.py
===============
Builds a single self-contained HTML report (Plotly, clickable legends) comparing
battery trading performance across forecasting methods.

Sections:
    1. Cumulative profit  (perfect foresight, QR, hard argmax, weighted, unanimous)
    2. Cumulative profit for the alpha-sweep (twostep alpha = 0.0 .. 0.5)
    3. Table: profit-per-trade metrics per forecasting method
    4. Boxplots of daily aggregated profit (+ mean markers)
    5. Conclusion (placeholder, fill in later)

Run:
    cd DA_Optimization_Git/src/analysis
    python build_report.py
Output:
    DA_Optimization_Git/reports/trading_report.html
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pcolors

from ip_results import load_results_parquet

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent                 # .../src/analysis
PROJECT = HERE.parents[1]                              # .../DA_Optimization_Git
RESULTS = PROJECT / "results" / "ip_rolling_ce"
OUT = PROJECT / "reports" / "trading_report.html"

PROFIT = ".profit_realized_eur"
ECH = ".e_ch_kwh"
EDIS = ".e_dis_kwh"
TRADE_EPS = 1e-6  # kWh: minimum activity to count a timestep as a trade


# --------------------------------------------------------------------------- #
# Method configuration
# --------------------------------------------------------------------------- #
# Main comparison (section 1)
MAIN_METHODS = [
    ("Perfect foresight", "ip_rolling_ce_qr_deterministic_perfect_foresight_TP_L2_0.01_CP_0__v1.parquet"),
    ("Quantile regression", "ip_rolling_ce_qr_deterministic_forecast_TP_L2_0.01_CP_0__v1.parquet"),
    ("Hard argmax", "ip_rolling_ce_twostep0p5_deterministic_forecast_TP_L2_0.01_CP_0__v1.parquet"),
    ("Weighted composite", "ip_rolling_ce_twostep0p0_deterministic_forecast_TP_L2_0.01_CP_0__v1.parquet"),
    ("Unanimous vote", "ip_rolling_ce_twostep_unanimous_deterministic_forecast_TP_L2_0.01_CP_0__v1.parquet"),
]

# Alpha sweep (section 2): alpha = 0.00, 0.05, ... 0.50
ALPHA_VALUES = ["0p0", "0p05", "0p1", "0p15", "0p2", "0p25", "0p3", "0p35", "0p4", "0p45", "0p5"]
ALPHA_METHODS = [
    (f"alpha={a.replace('p', '.')}",
     f"ip_rolling_ce_twostep{a}_deterministic_forecast_TP_L2_0.01_CP_0__v1.parquet")
    for a in ALPHA_VALUES
]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _detect_prefix(df: pd.DataFrame) -> str:
    """Strip '.profit_realized_eur' to recover the strategy prefix."""
    for c in df.columns:
        if str(c).endswith(PROFIT):
            return str(c)[: -len(PROFIT)]
    raise KeyError(f"No '*{PROFIT}' column. Columns: {list(df.columns)[:10]}")


def load_methods(methods):
    """Return list of dicts with label, profit/charge/discharge series."""
    loaded = []
    for label, fname in methods:
        path = RESULTS / fname
        if not path.exists():
            print(f"  [skip] missing file for '{label}': {fname}")
            continue
        df, meta = load_results_parquet(path)
        pref = _detect_prefix(df)
        profit = pd.to_numeric(df[f"{pref}{PROFIT}"], errors="coerce").fillna(0.0)
        ch = pd.to_numeric(df[f"{pref}{ECH}"], errors="coerce").fillna(0.0)
        dis = pd.to_numeric(df[f"{pref}{EDIS}"], errors="coerce").fillna(0.0)
        loaded.append({
            "label": label,
            "prefix": pref,
            "profit": profit,
            "ch": ch,
            "dis": dis,
        })
        print(f"  [ok]   {label:<22} ({pref})")
    return loaded


# --------------------------------------------------------------------------- #
# Section 1 & 2: cumulative profit
# --------------------------------------------------------------------------- #
def fig_cumulative(methods, title):
    fig = go.Figure()
    for m in methods:
        cum = m["profit"].cumsum()
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values, mode="lines", name=m["label"],
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>€%{y:,.0f}<extra>" + m["label"] + "</extra>",
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Cumulative realized profit [€]",
        legend_title="Method (click to toggle)",
        hovermode="x unified",
        template="plotly_white",
        height=520,
    )
    return fig


# --------------------------------------------------------------------------- #
# Section 3: profit-per-trade table
# --------------------------------------------------------------------------- #
def table_profit_per_trade(methods):
    rows = []
    for m in methods:
        total_profit = float(m["profit"].sum())
        volume_mwh = float((m["ch"] + m["dis"]).sum()) / 1000.0
        activity = (m["ch"] + m["dis"]) > TRADE_EPS
        n_trades = int(activity.sum())
        rows.append({
            "Method": m["label"],
            "Total profit [€]": total_profit,
            "Traded volume [MWh]": volume_mwh,
            "# trades": n_trades,
            "Profit per trade [€]": total_profit / n_trades if n_trades else float("nan"),
            "Profit per MWh [€]": total_profit / volume_mwh if volume_mwh else float("nan"),
        })
    df = pd.DataFrame(rows).set_index("Method")
    return df


def table_to_html(df: pd.DataFrame) -> str:
    fmt = {
        "Total profit [€]": "{:,.0f}",
        "Traded volume [MWh]": "{:,.1f}",
        "# trades": "{:,d}",
        "Profit per trade [€]": "{:,.2f}",
        "Profit per MWh [€]": "{:,.2f}",
    }
    styled = df.style.format(fmt).set_table_attributes('class="report-table"')
    try:
        return styled.to_html()
    except AttributeError:  # older pandas
        return df.to_html(classes="report-table", float_format=lambda x: f"{x:,.2f}")


# --------------------------------------------------------------------------- #
# Section 4: boxplots of daily aggregated profit
# --------------------------------------------------------------------------- #
def fig_daily_boxplots(methods):
    fig = go.Figure()
    palette = pcolors.qualitative.Plotly
    means_x, means_y = [], []
    for i, m in enumerate(methods):
        daily = m["profit"].groupby(pd.Grouper(freq="D")).sum()
        daily = daily[daily.index.notna()]
        color = palette[i % len(palette)]
        fig.add_trace(go.Box(
            y=daily.values,
            name=m["label"],
            boxpoints="outliers",
            marker_color=color,
            line_color=color,
            hovertemplate="€%{y:,.0f}<extra>" + m["label"] + "</extra>",
        ))
        means_x.append(m["label"])
        means_y.append(float(daily.mean()))
    # explicit mean markers overlaid on the boxes
    fig.add_trace(go.Scatter(
        x=means_x, y=means_y, mode="markers", name="Mean daily profit",
        marker=dict(symbol="diamond", size=11, color="black",
                    line=dict(width=1, color="white")),
        hovertemplate="Mean €%{y:,.1f}<extra>%{x}</extra>",
    ))
    fig.update_layout(
        title="Daily aggregated profit distribution (risk on a bad day)",
        yaxis_title="Daily realized profit [€]",
        xaxis_title="Method",
        legend_title="Click to toggle",
        template="plotly_white",
        height=560,
    )
    return fig


# --------------------------------------------------------------------------- #
# HTML assembly
# --------------------------------------------------------------------------- #
CSS = """
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         margin: 0 auto; max-width: 1180px; padding: 32px; color: #1a1a1a; }
  h1 { border-bottom: 3px solid #2c3e50; padding-bottom: 12px; }
  h2 { color: #2c3e50; margin-top: 48px; border-left: 5px solid #3498db;
       padding-left: 12px; }
  .subtitle { color: #666; font-size: 0.95em; }
  .report-table { border-collapse: collapse; width: 100%; margin: 16px 0;
                  font-size: 0.95em; }
  .report-table th, .report-table td { border: 1px solid #ddd; padding: 8px 12px;
                  text-align: right; }
  .report-table th { background: #2c3e50; color: #fff; }
  .report-table tr:nth-child(even) { background: #f7f9fb; }
  .report-table td:first-child, .report-table th:first-child { text-align: left; }
  .note { background: #fff8e1; border-left: 4px solid #f1c40f; padding: 12px 16px;
          border-radius: 4px; }
</style>
"""

CONCLUSION_PLACEHOLDER = """
<p>In this report, we compared the financial results when using different
forecasting methods in our MPC approach. Unfortunately, my intuition was wrong,
and the fundamentally different forecasting approach did not provide any
noticeable benefit in the trading case studies.</p>

<p>This brings me back to the start of this idea. I believed that a two-step
forecast would be beneficial, because from my work at Gridual I know that it can
outperform regular regressions for very short lead times in the order of 1 to 10
minutes. With a minimum lead time of 15 minutes, the advantage of the two-step
method is not yet noticeable. In addition, Elia's per-minute signal is also very
accurate for these lead times, and it outperforms our neural network at Gridual
until a lead time of 8/9 minutes. The per-minute signal is based on the imbalance
price formula, which is very similar to the two-step forecast (it is built based
on the SI volume).</p>

<p>Therefore, I think the value of the classifiers only becomes clear as we also
include intra-QH information in the forecasting process. In any case, I believe
this would have been the natural follow-up study, so I think this would be a good
direction for this research.</p>

<p>With the new server, I have managed to decimate the NN training time, so I'm
optimistic in terms of the implementation time of this forecasting granularity
change. I'd be happy to hear your thoughts!</p>
"""


def fig_html(fig, first):
    return fig.to_html(full_html=False,
                       include_plotlyjs="cdn" if first else False)


def build():
    print("Loading main methods...")
    main = load_methods(MAIN_METHODS)
    print("Loading alpha sweep...")
    sweep = load_methods(ALPHA_METHODS)

    f1 = fig_cumulative(main, "Cumulative profit by forecasting method")
    f2 = fig_cumulative(sweep, "Cumulative profit — alpha sweep (weighted ↔ hard threshold)")
    tbl = table_to_html(table_profit_per_trade(main))
    f4 = fig_daily_boxplots(main)

    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Battery trading — forecasting comparison</title>",
        CSS, "</head><body>",
        "<h1>Battery Trading: Imbalance-Price Forecasting Comparison</h1>",
        "<p class='subtitle'>Test window 2023-01-01 → 2024-01-01 · "
        "15-min resolution · realized profit from rolling certainty-equivalent MPC. "
        "Legends are clickable — click an entry to hide/show a series.</p>",

        "<h2>1. Cumulative profit by method</h2>",
        fig_html(f1, first=True),

        "<h2>2. Cumulative profit — alpha sweep</h2>",
        "<p class='subtitle'>twostep composite: alpha=0.0 is fully weighted, "
        "alpha=0.5 is the plain hard threshold.</p>",
        fig_html(f2, first=False),

        "<h2>3. Profit per trade</h2>",
        "<p class='subtitle'>A trade = any 15-min step with battery activity "
        "(charge or discharge &gt; 0).</p>",
        tbl,

        "<h2>4. Daily aggregated profit (risk view)</h2>",
        "<p class='subtitle'>Box = daily profit distribution; black diamond = mean "
        "daily profit. Wide lower whiskers indicate worse bad-day risk.</p>",
        fig_html(f4, first=False),

        "<h2>5. Conclusion</h2>",
        CONCLUSION_PLACEHOLDER,

        "</body></html>",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"\nReport written: {OUT}")


if __name__ == "__main__":
    build()