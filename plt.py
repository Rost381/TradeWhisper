import pandas as pd
import os 
import numpy as np
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpdates
import matplotlib.dates as mdates
from dotenv import load_dotenv
from indicacators_sets import (
    calculate_fractals,
    find_sr,
    find_sr_lines,
)


load_dotenv(".env-plot")
CANDLESTICK_FILE = os.getenv("CANDLESTICK_FILE", "futures_data_5m/DOGEUSDT.csv")

SHOW_BALANCE = False
SHOW_PAEKS = True
SHOW_PAEKS_LINE = True
SHOW_LEVELS = False
SHOW_FRACTALS = False

df = pd.read_csv(CANDLESTICK_FILE)

df=df[2980:5860]


# convert into timestamptime object
df["timestamp"] = pd.to_datetime(df["timestamp"])

# apply map function
df["timestamp"] = df["timestamp"].map(mpdates.date2num)

# creating Subplots
fig, ax = plt.subplots(figsize=(23, 12))

# plotting the data
candlestick_ohlc(
    ax, df.values, width=0.0025, colorup="green", colordown="red", alpha=0.8
)

# Calculate Peaks and Levels
if SHOW_PAEKS or SHOW_LEVELS:
    peaks, troughs, support_levels, resistance_levels = find_sr(df)

    # print(support_levels)
    print(df[39:154])

# Draw Peaks
if SHOW_PAEKS:
    plt.scatter(
        df["timestamp"].iloc[peaks],
        df["high"].iloc[peaks],
        color="red",
        label="Сопротивление",
    )
    plt.scatter(
        df["timestamp"].iloc[troughs],
        df["low"].iloc[troughs],
        color="green",
        label="Поддержка",
    )

# Draw Fractals
if SHOW_FRACTALS:
    # Отображение фракталов
    df = calculate_fractals(df, window=5)  # 5 свечей
    
    for _, row in df.iterrows():
        if not np.isnan(row["fractal_high"]):
            ax.annotate(
                "↑",
                xy=(row["timestamp"], row["fractal_high"]),
                color="green",
                fontsize=12,
                ha="center",
            )
        if not np.isnan(row["fractal_low"]):
            ax.annotate(
                "↓",
                xy=(row["timestamp"], row["fractal_low"]),
                color="red",
                fontsize=12,
                ha="center",
            )

if SHOW_PAEKS_LINE:
    df = find_sr_lines(df, dist=48)
    # Отображение фракталов
    for _, row in df.iterrows():
        if not np.isnan(row["resistance"]):
            ax.annotate(
                "-",
                xy=(row["timestamp"], row["resistance"]),
                color="red",
                fontsize=12,
                ha="center",
            )
        if not np.isnan(row["support"]):
            ax.annotate(
                "-",
                xy=(row["timestamp"], row["support"]),
                color="green",
                fontsize=12,
                ha="center",
            )

# allow grid
ax.grid(True)

# Setting labels
ax.set_xlabel("Time")
ax.set_ylabel("Price")

# setting title
plt.title("Prices")

# Formatting Date
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m.%d %H:%M"))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())


fig.autofmt_xdate()
fig.tight_layout()

plt.legend()
# show the plot
plt.show()
