# import pandas as pd
# import matplotlib.pyplot as plt
#
# root = './Data/EPEXData/'
#
# ### Auctions
#
# ## Day-ahead
# # Hourly
# da_auction_vol = pd.read_csv(root+'Day-Ahead Auction/Hourly/Historical/Prices_Volumes/auction_spot_volumes_belgium_2025.csv')
# da_auction_price = pd.read_csv(root+'Day-Ahead Auction/Hourly/Historical/Prices_Volumes/auction_spot_prices_belgium_2025.csv')
#
# plt.plot(da_auction_vol)
# plt.show()
#
# plt.plot(da_auction_price)
# plt.show()
#
# ## Intraday
#
#
#
# ### Continuous
#
# ## Intraday

from pathlib import Path
import zipfile
import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go
import plotly.io as pio

# ----------------- CONFIG -----------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data/EPEXData/Intraday Continuous/EOD/Historical/Transactions/" # Continuous_Trades-BE-2025"

TZ = "Europe/Brussels"
BIN_MINUTES = 5 # trade-time bin size for x-axis
WINDOW_HOURS_BEFORE = 48 # show trades up to 24h before delivery start (per product)
EXCLUDE_SELF_TRADES = False # optional: set True to drop SelfTrade == 'Y' if present

# New filters:
HIDE_BLOCK_PRODUCTS = False # hide delivery products longer than MAX_STANDARD_MINUTES
MAX_STANDARD_MINUTES = 60 # treat >60min as block product (keeps 15-min + hourly products)
MIN_TRADES_PER_BIN = 1 # only show (DeliveryStart, TradeBin) buckets with >= this many trades

# ----------------- LOAD TRADES FROM ZIPs -----------------

rows = []
zip_paths = list(DATA_DIR.rglob("*.zip"))
# zip_paths = [DATA_DIR]
assert zip_paths, f"No .zip files found under {DATA_DIR}"

for zp in zip_paths:
    with zipfile.ZipFile(zp, "r") as zf:
        # only Continuous_Trades for BE (executed trades), not Orders
        members = [
        n for n in zf.namelist()
        if (".csv" in n) and ("Continuous_Trades-BE-2025" in Path(n).name)
        ]
        print(members)
        for name in members:
            with zf.open(name) as f:
                dfi = pd.read_csv(f, comment="#", sep=",", encoding="latin1", engine="python", on_bad_lines="warn")
                dfi["__zip"] = zp.name
                dfi["__member"] = name
                rows.append(dfi)

trades = pd.concat(rows, ignore_index=True)

# Required columns per spec
required = ["DeliveryStart", "DeliveryEnd", "ExecutionTime", "DeliveryArea", "Price", "Volume", "VolumeUnit", "Currency"]
missing = [c for c in required if c not in trades.columns]
assert not missing, f"Missing required columns: {missing}"

# Keep BE legs only
trades = trades[trades["DeliveryArea"] == "BE"].copy()

# Optional self-trade filter
if EXCLUDE_SELF_TRADES and "SelfTrade" in trades.columns:
    trades = trades[trades["SelfTrade"].astype(str).str.upper() != "Y"].copy()

# ----------------- TIME PARSING (UTC FIRST, DST-SAFE BINNING) -----------------

# Parse as UTC first (keep UTC columns)
trades["ExecutionTime_UTC"] = pd.to_datetime(trades["ExecutionTime"], utc=True)
trades["DeliveryStart_UTC"] = pd.to_datetime(trades["DeliveryStart"], utc=True)
trades["DeliveryEnd_UTC"] = pd.to_datetime(trades["DeliveryEnd"], utc=True)

# Bin in UTC (no DST issues)
trades["TradeBin_UTC"] = trades["ExecutionTime_UTC"].dt.floor(f"{BIN_MINUTES}min")

# Convert to BE for plotting/labels
trades["DeliveryStart"] = trades["DeliveryStart_UTC"].dt.tz_convert(TZ)
trades["DeliveryEnd"] = trades["DeliveryEnd_UTC"].dt.tz_convert(TZ)
trades["ExecutionTime"] = trades["ExecutionTime_UTC"].dt.tz_convert(TZ)
trades["TradeBin"] = trades["TradeBin_UTC"].dt.tz_convert(TZ)

# ----------------- PRODUCT FILTERS (BLOCK vs STANDARD) -----------------

# Duration in minutes computed in UTC (avoid DST quirks)
trades["ProductMinutes"] = (trades["DeliveryEnd_UTC"] - trades["DeliveryStart_UTC"]).dt.total_seconds() / 60.0

if HIDE_BLOCK_PRODUCTS:
    trades = trades[trades["ProductMinutes"] <= MAX_STANDARD_MINUTES].copy()

# ----------------- NUMERIC CLEANING -----------------

trades["Price"] = pd.to_numeric(trades["Price"], errors="coerce")
trades["Volume"] = pd.to_numeric(trades["Volume"], errors="coerce")
trades = trades.dropna(subset=["Price", "Volume", "DeliveryStart", "ExecutionTime"])

# Delivery day (Belgium local day of delivery start)
trades["DeliveryDate"] = trades["DeliveryStart"].dt.date

# ----------------- MATRIX BUILDING -----------------

def product_label(start, end):
    return f"{start.strftime('%H:%M')}–{end.strftime('%H:%M')}"

def build_matrix_for_date(delivery_date):
    ddf = trades[trades["DeliveryDate"] == delivery_date].copy()

    # Trades in the time window prior to delivery start
    ddf = ddf[
    (ddf["ExecutionTime"] >= ddf["DeliveryStart"] - timedelta(hours=WINDOW_HOURS_BEFORE)) &
    (ddf["ExecutionTime"] <= ddf["DeliveryStart"])
    ]

    # VWAP within each (product, tradebin): sum(price*vol)/sum(vol)
    ddf["pxv"] = ddf["Price"] * ddf["Volume"]

    grp = ddf.groupby(["DeliveryStart", "TradeBin"], as_index=False).agg(
    pxv=("pxv", "sum"),
    vol=("Volume", "sum"),
    ntrades=("Price", "size"),
    )

    # Keep only sufficiently liquid bins
    grp = grp[grp["ntrades"] >= MIN_TRADES_PER_BIN].copy()

    grp["vwap"] = grp["pxv"] / grp["vol"]

    mat = grp.pivot(index="DeliveryStart", columns="TradeBin", values="vwap")
    mat = mat.sort_index(ascending=False) # latest delivery at top
    mat = mat.reindex(sorted(mat.columns), axis=1) # time left->right

    # y labels
    ends = (
    ddf[["DeliveryStart", "DeliveryEnd"]]
    .drop_duplicates("DeliveryStart", keep="last")
    .set_index("DeliveryStart")
    .reindex(mat.index)["DeliveryEnd"]
    )
    y = [product_label(s, e) for s, e in zip(mat.index, ends)]
    x = list(mat.columns)
    z = mat.to_numpy()
    return x, y, z

def make_fig(delivery_date):
    x, y, z = build_matrix_for_date(delivery_date)
    fig = go.Figure(go.Heatmap(
    x=x, y=y, z=z,
    colorbar=dict(title="VWAP [€/MWh]"),
    hovertemplate="Trade bin: %{x}<br>Product: %{y}<br>VWAP: %{z:.2f} €/MWh<extra></extra>"
    ))
    fig.update_layout(
    title=(
    f"BE Continuous Intraday – delivery {delivery_date} "
    f"(VWAP per {BIN_MINUTES}min bin; prior {WINDOW_HOURS_BEFORE}h; "
    f"min {MIN_TRADES_PER_BIN} trades/bin; "
    f"{'no blocks' if HIDE_BLOCK_PRODUCTS else 'blocks included'})"
    ),
    height=820,
    xaxis_title="Execution time (BE)",
    yaxis_title="Delivery product",
    )
    return fig

# ----------------- INTERACTIVE DROPDOWN -----------------

dates = sorted(trades["DeliveryDate"].unique())
assert dates, "No delivery dates found after filtering."

fig = make_fig(dates[0])

buttons = []
for d in dates:
    x, y, z = build_matrix_for_date(d)
    buttons.append(dict(
    label=str(d),
    method="update",
    args=[
    {"x": [x], "y": [y], "z": [z]},
    {"title": (
    f"BE Continuous Intraday – delivery {d} "
    f"(VWAP per {BIN_MINUTES}min bin; prior {WINDOW_HOURS_BEFORE}h; "
    f"min {MIN_TRADES_PER_BIN} trades/bin; "
    f"{'no blocks' if HIDE_BLOCK_PRODUCTS else 'blocks included'})"
    )}
    ],
    ))

    fig.update_layout(
    updatemenus=[dict(buttons=buttons, direction="down", showactive=True, x=0.01, y=1.12)]
    )

    pio.renderers.default = "browser"
    fig.show()
