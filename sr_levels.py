import os
import pandas as pd
from dotenv import load_dotenv
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt

# Загрузка переменных окружения
load_dotenv()
CANDLESTICK_FILE = os.getenv("CANDLESTICK_FILE", "futures_data_5m/DOGEUSDT.csv")

df = pd.read_csv(CANDLESTICK_FILE, parse_dates=["timestamp"])

df=df[2980:5860]

# Преобразование столбцов в нужные типы данных
df["Open time"] = df["timestamp"]  # pd.to_datetime(df["timestamp"], unit="ms")
# df["Close time"] = pd.to_datetime(df["timestamp"].shift(-1), unit="ms")
numeric_cols = ["open", "high", "low", "close", "volume"]
df[numeric_cols] = df[numeric_cols].astype(float)

# print(df)


def find_price_peaks(df, dist=48):
    # Найдем пики (локальные максимумы)
    peaks, _ = find_peaks(df["high"], distance=dist)
    # Найдем впадины (локальные минимумы)
    troughs, _ = find_peaks(-df["low"], distance=dist)
    support_levels = df["low"].iloc[troughs]
    resistance_levels = df["high"].iloc[peaks]
    return peaks, troughs, support_levels, resistance_levels

def cluster_levels(levels, price_threshold=500):
    clustered_levels = []
    levels = sorted(levels)
    cluster = [levels[0]]

    for level in levels[1:]:
        if abs(level - np.mean(cluster)) <= price_threshold:
            cluster.append(level)
        else:
            clustered_levels.append(np.mean(cluster))
            cluster = [level]
    clustered_levels.append(np.mean(cluster))
    return clustered_levels


# Экран

plt.figure(figsize=(23, 12))

# Линия
plt.plot(df["Open time"], df["close"], label="Цена закрытия")


peaks, troughs, support_levels, resistance_levels = find_price_peaks(df)

plt.scatter(
    df["Open time"].iloc[peaks],
    df["high"].iloc[peaks],
    color="red",
    label="Сопротивление",
)
plt.scatter(
    df["Open time"].iloc[troughs],
    df["low"].iloc[troughs],
    color="green",
    label="Поддержка",
)

price_threshold = 1  # Настройте в зависимости от волатильности
clustered_support = cluster_levels(support_levels, price_threshold)
clustered_resistance = cluster_levels(resistance_levels, price_threshold)

for level in clustered_support:
    plt.hlines(
        level,
        df["Open time"].min(),
        df["Open time"].max(),
        colors="green",
        linestyles="dashed",
    )
for level in clustered_resistance:
    plt.hlines(
        level,
        df["Open time"].min(),
        df["Open time"].max(),
        colors="red",
        linestyles="dashed",
    )

plt.legend()
plt.show()
