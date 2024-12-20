import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.patches import Rectangle
from datetime import timedelta
from ta.trend import EMAIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()
CANDLESTICK_FILE = os.getenv("CANDLESTICK_FILE", "futures_data_5m/DOGEUSDT.csv")
FONT_SIZE = int(os.getenv("FONT_SIZE", 9))

# Загрузка данных
candlesticks = pd.read_csv(CANDLESTICK_FILE, parse_dates=["timestamp"])


# Функция для расчёта Pivot Points
def calculate_pivot_points(df, window):
    rolling = df.rolling(window=window)
    pivot = (rolling["high"].max() + rolling["low"].min() + rolling["close"].mean()) / 3
    r1 = 2 * pivot - rolling["low"].min()
    s1 = 2 * pivot - rolling["high"].max()
    r2 = pivot + (rolling["high"].max() - rolling["low"].min())
    s2 = pivot - (rolling["high"].max() - rolling["low"].min())
    return pivot, r1, s1, r2, s2


# Функция для расчёта фракталов
def calculate_fractals(df, window):
    df["fractal_high"] = (
        df["high"]
        .rolling(window=window, center=True)
        .apply(
            lambda x: x[window // 2] if x[window // 2] == max(x) else np.nan, raw=True
        )
    )
    df["fractal_low"] = (
        df["low"]
        .rolling(window=window, center=True)
        .apply(
            lambda x: x[window // 2] if x[window // 2] == min(x) else np.nan, raw=True
        )
    )
    return df


# Расчёт уровней для 5-минутного таймфрейма (Pivot Points и фракталы)
(
    candlesticks["pivot"],
    candlesticks["r1"],
    candlesticks["s1"],
    candlesticks["r2"],
    candlesticks["s2"],
) = calculate_pivot_points(
    candlesticks, window=288
)  # Часовой период
candlesticks = calculate_fractals(candlesticks, window=5)  # 5 свечей


# Уменьшение плотности линий старшего таймфрейма
# step = len(candlesticks) // 100  # Примерное упрощение
# if step > 1:
#     candlesticks = candlesticks.iloc[::step]
# Агрегация данных для старшего таймфрейма (1H)
agg_data = (
    candlesticks.resample("1h", on="timestamp")
    .agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    .dropna()
)
agg_data["pivot"], agg_data["r1"], agg_data["s1"], agg_data["r2"], agg_data["s2"] = (
    calculate_pivot_points(agg_data, window=24)
)  # Недельный период (24 часа)


# Присоединение уровней старшего таймфрейма к основному DataFrame
candlesticks["higher_pivot"] = agg_data["pivot"].reindex(
    candlesticks["timestamp"], method="ffill"
)
candlesticks["higher_r1"] = agg_data["r1"].reindex(
    candlesticks["timestamp"], method="ffill"
)
candlesticks["higher_s1"] = agg_data["s1"].reindex(
    candlesticks["timestamp"], method="ffill"
)

candlesticks["bollinger_hband"] = BollingerBands(
    candlesticks["close"], window=20, window_dev=2
).bollinger_hband()
candlesticks["bollinger_lband"] = BollingerBands(
    candlesticks["close"], window=20, window_dev=2
).bollinger_lband()

#candlesticks[["fractal_high", "fractal_low"]] = candlesticks[["fractal_high", "fractal_low"]].fillna(method="ffill")
# print(candlesticks)
#print(candlesticks[["fractal_high", "fractal_low"]])
#input("Continue?")

# Настройка графика
fig, ax1 = plt.subplots(figsize=(16, 8))
ax2 = ax1.twinx()
plt.rcParams.update({"font.size": FONT_SIZE})

# Форматирование дат на оси X
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m.%d %H:%M"))
ax1.xaxis.set_major_locator(mdates.AutoDateLocator())

# Отображение свечей
candlestick_rects = []
for _, row in candlesticks.iterrows():
    color = "green" if row["close"] > row["open"] else "red"
    ax1.plot(
        [row["timestamp"], row["timestamp"]],
        [row["low"], row["high"]],
        color="black",
        linewidth=0.5,
    )
    rect = Rectangle(
        (mdates.date2num(row["timestamp"]) - 0.001, min(row["open"], row["close"])),
        0.002,
        abs(row["close"] - row["open"]),
        color=color,
    )
    candlestick_rects.append(rect)
    ax1.add_patch(rect)

# Отображение BollingerBands
# ax1.plot(
#     candlesticks["timestamp"],
#     candlesticks["bollinger_hband"],
#     label="HB",
#     color="blue",
#     linestyle="-",
# )
# ax1.plot(
#     candlesticks["timestamp"],
#     candlesticks["bollinger_lband"],
#     label="LB",
#     color="green",
#     linestyle="-",
# )

# Отображение уровней Pivot Points
# ax1.plot(
#     candlesticks["timestamp"],
#     candlesticks["pivot"],
#     label="Pivot",
#     color="blue",
#     linestyle="--",
# )
# ax1.plot(
#     candlesticks["timestamp"],
#     candlesticks["r1"],
#     label="R1",
#     color="green",
#     linestyle="--",
# )
# ax1.plot(
#     candlesticks["timestamp"],
#     candlesticks["s1"],
#     label="S1",
#     color="red",
#     linestyle="--",
# )

# ax1.plot(
#     candlesticks["timestamp"],
#     candlesticks["r2"],
#     label="R2",
#     color="darkgreen",
#     linestyle="--",
# )
# ax1.plot(
#     candlesticks["timestamp"],
#     candlesticks["s2"],
#     label="S2",
#     color="darkred",
#     linestyle="--",
# )

# Отображение уровней старшего таймфрейма
# ax1.plot(
#     candlesticks["timestamp"],
#     candlesticks["higher_pivot"],
#     label="Higher Pivot",
#     color="purple",
#     linestyle="-",
# )
# ax1.plot(
#     candlesticks["timestamp"],
#     candlesticks["higher_r1"],
#     label="Higher R1",
#     color="orange",
#     linestyle="-",
# )
# ax1.plot(
#     candlesticks["timestamp"],
#     candlesticks["higher_s1"],
#     label="Higher S1",
#     color="brown",
#     linestyle="-",
# )

# Отображение фракталов
# for _, row in candlesticks.iterrows():
#     if not np.isnan(row["fractal_high"]):
#         ax1.annotate(
#             "▼",
#             xy=(row["timestamp"], row["fractal_high"]),
#             color="gray",
#             fontsize=12,
#             ha="center",
#         )
#     if not np.isnan(row["fractal_low"]):
#         ax1.annotate(
#             "▲",
#             xy=(row["timestamp"], row["fractal_low"]),
#             color="gray",
#             fontsize=12,
#             ha="center",
#         )

# Подписи осей и легенда
ax1.set_xlabel("Время")
ax1.set_ylabel("Цена")
ax1.legend()

# Отображение графика
fig.tight_layout()
plt.show()
