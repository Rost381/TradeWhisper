import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
matplotlib.use('TkAgg') 

from mplfinance.original_flavor import candlestick_ohlc
import mplcursors
from dotenv import load_dotenv
import argparse



# Загрузка переменных окружения
load_dotenv()
load_dotenv(".env-bk")

MODEL_SUFFIX = os.getenv("MODEL_SUFFIX")
STATS_DIR = os.getenv("STATS_DIR")
TIMEFRAME = os.getenv("TIMEFRAME")
DATA_DIR = os.getenv("DATA_DIR")

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

model_suffix_cmd= args.suffix
if model_suffix_cmd:
    MODEL_SUFFIX = model_suffix_cmd

STATS_DIR = f"{STATS_DIR}_{MODEL_SUFFIX}"

def plot_result(result_trades_path, plt_show=None):
    # Загрузка данных
    candlesticks_paht = f"{DATA_DIR}_{TIMEFRAME}/DOGEUSDT.csv"
    candlesticks = pd.read_csv(candlesticks_paht, parse_dates=["timestamp"])
    trades = pd.read_csv(result_trades_path, parse_dates=["open_ts", "close_ts"])

    # Ограничение данных по диапазону из файла сделок
    start_time = trades["open_ts"].min()
    end_time = trades["close_ts"].max()
    if end_time < trades["open_ts"].max():
        end_time = trades["open_ts"].max()
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

    try:
        plot_period = f"{result_trades_path.split('/')[1].split(".")[0].split("_")[1].upper()}"
    except:
        plot_period = "DEFAULT"

    # Подписи осей
    
    plt.title(f'Chart for {MODEL_SUFFIX} from {start_time} to {end_time}. Period: {plot_period}', fontsize=14)
    ax1.set_xlabel("Время")
    ax1.set_ylabel("Цена")
    ax2.set_ylabel("Баланс")


    # Отображение графика
    fig.tight_layout()
    
    plot_file = f"{result_trades_path.split('.')[0]}.png"
    
    if plt_show == "show":
        print("show plot")
        plt.show()
        plt.close
        return
    elif plt_show == "save":
        plt.savefig(plot_file, format='png', dpi=800)  # Сохранение с высоким разрешением
        print(f"Файл изображения сохранен в {plot_file}")
        plt.close
        return
    else:
        plt.savefig(result_trades_path, format='png', dpi=800)  # Сохранение с высоким разрешением
        print(f"Файл изображения сохранен в {plot_file}")
        plt.show()
        plt.close
        return
    
if __name__ == "__main__":
    
    result_trades_path = f"{STATS_DIR}/result.csv"
    plot_result(result_trades_path, plt_show="show")