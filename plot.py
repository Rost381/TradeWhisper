import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from mplfinance.original_flavor import candlestick_ohlc
import mplcursors
from dotenv import load_dotenv
import argparse


# Загрузка переменных окружения
load_dotenv(".env-bk")

MODEL_SUFFIX = os.getenv("MODEL_SUFFIX")
STATS_DIR = os.getenv("STATS_DIR")
TIMEFRAME = os.getenv("TIMEFRAME")
BK_DIR = os.getenv("BK_DIR")

# STATS_DIR = f"{STATS_DIR}_{MODEL_SUFFIX}"


FONT_SIZE = int(os.getenv("FONT_SIZE", 9))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Script for processing candlestick and trade data."
    )
    parser.add_argument(
        "suffix",
        nargs="?",
        type=str,
        default="",
        help="Suffix for the model (MODEL_SUFFIX).",
    )
    return parser.parse_args()

args = parse_arguments()

CANDLESTICK_FILE = f"{BK_DIR}_{TIMEFRAME}/DOGEUSDT.csv"

MODEL_SUFFIX = args.suffix

if MODEL_SUFFIX:
    STATS_DIR = f"{STATS_DIR}_{MODEL_SUFFIX}"

TRADES_FILE = f"{STATS_DIR}/result.csv"


STATS_DIR = f"{STATS_DIR}_{MODEL_SUFFIX}"
# Загрузка данных
candlesticks = pd.read_csv(CANDLESTICK_FILE, parse_dates=["timestamp"])
trades = pd.read_csv(TRADES_FILE, parse_dates=["open_ts", "close_ts"])

# Ограничение данных по диапазону из файла сделок
start_time = trades["open_ts"].min()
end_time = trades["close_ts"].max()
candlesticks = candlesticks[
    (candlesticks["timestamp"] >= start_time) & (candlesticks["timestamp"] <= end_time)
]

# Масштабирование баланса
price_min = candlesticks[["low"]].min().values[0]
price_max = candlesticks[["high"]].max().values[0]
trades["scaled_balance"] = (
    trades["balance"]
    * (price_max - price_min)
    / (trades["balance"].max() - trades["balance"].min())
    + price_min
)


# Настройка графика
fig, ax1 = plt.subplots(figsize=(16, 8))
ax2 = ax1.twinx()
plt.rcParams.update({"font.size": FONT_SIZE})

# Форматирование дат на оси X
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m.%d %H:%M"))
ax1.xaxis.set_major_locator(mdates.AutoDateLocator())

# Отображение свечей
df = candlesticks.copy()
# convert into timestamptime object
df["timestamp"] = pd.to_datetime(df["timestamp"])

# apply map function
df["timestamp"] = df["timestamp"].map(mdates.date2num)
candlestick_ohlc(
    ax1, df.values, width=0.0025, colorup="green", colordown="red", alpha=0.8
)
# candlestick_rects = []
# for _, row in candlesticks.iterrows():
#     color = "green" if row["close"] > row["open"] else "red"
#     ax1.plot(
#         [row["timestamp"], row["timestamp"]],
#         [row["low"], row["high"]],
#         color="black",
#         linewidth=0.5,
#     )
#     rect = plt.Rectangle(
#         (mdates.date2num(row["timestamp"]) - 0.001, min(row["open"], row["close"])),
#         0.002,
#         abs(row["close"] - row["open"]),
#         color=color,
#     )
#     candlestick_rects.append(rect)
#     ax1.add_patch(rect)

# Отображение сделок
trade_points = []
for _, trade in trades.iterrows():
    if trade["type"] == "short":
        color = "black" #"blue"
        marker_open = "v"
    else:
        color = "blue" #"purple"
        marker_open = "^"
    point_open = ax1.scatter(
        trade["open_ts"], trade["open_price"], color=color, marker=marker_open, zorder=5
    )
    point_close = ax1.scatter(
        trade["close_ts"], trade["close_price"], color=color, marker="X", zorder=5
    )
    trade_points.extend([point_open, point_close])

# Отображение баланса
ax2.plot(
    trades["close_ts"],
    trades["balance"],
    color="purple",
    label="Баланс",
    alpha=0.7,
)


# Отображение сделок на линии баланса
for _, trade in trades.iterrows():
    balance_value = trade["balance"]
    if trade["type"] == "short":
        color = "red"
        marker_open = "x"
    else:
        color = "green"
        marker_open = "x"
    ax2.scatter(
        trade["close_ts"], balance_value, color=color, marker=marker_open, zorder=5
    )


# Подписи осей
ax1.set_xlabel("Время")
ax1.set_ylabel("Цена")
ax2.set_ylabel("Баланс")

# Настройка интерактивности
# cursor = mplcursors.cursor(candlestick_rects + trade_points, hover=True)


# @cursor.connect("add")
# def on_add(sel):
#     if sel.artist in candlestick_rects:
#         idx = candlestick_rects.index(sel.artist)
#         row = candlesticks.iloc[idx]
#         sel.annotation.set_text(
#             f"Цена открытия: {row['open']:.5f}\nЦена закрытия: {row['close']:.5f}"
#         )
#     elif sel.artist in trade_points:
#         sel.annotation.set_text("Сделка")


# Отображение графика
fig.tight_layout()
plt.show()
