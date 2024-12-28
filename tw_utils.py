# Utilities for working with a exchange and others
import sys
import os
import asyncio
import logging
import argparse
import logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv


load_dotenv()


DATA_DIR = os.getenv("DATA_DIR", "futures_data")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Script for models processing."
    )
    parser.add_argument(
        "suffix",
        nargs="?",
        type=str,
        default="",
        help="Suffix for the model (MODEL_SUFFIX).",
    )
    return parser.parse_args()

async def is_continue(exchange, exit=False):
    """For correct exit. For Debug only"""
    if exit:
        user_choice = "n"
    else:
        user_choice = input("Продолжить ? (Y/n): ")
    if user_choice in ["n", "N", "т", "Т"]:
        if exchange:
            try:
                for task in asyncio.all_tasks():
                    task.cancel()
                await exchange.close()
            except:
                pass
        sys.exit(0)


def create_file_path(symbol, timeframe, data_dir=DATA_DIR) -> str:  # type: ignore
    # Формирование имени файла
    symbol_filename = symbol.split("/")
    symbol_filename = symbol_filename[0] + symbol_filename[1][:4] + ".csv"
    return f"{data_dir}_{timeframe}/{symbol_filename}"


# New fn with save df
async def get_full_data(exchange, symbol, timeframe=None, since=None, limit=None, run_mode=None):
    return_limit = limit  # Количество возврашаемых свечей
    all_ohlcv = []    
    file_path = create_file_path(symbol, timeframe)
    if run_mode == "BK":
        logging.info(f"Режим Бэктеста {symbol}")
        if os.path.exists(file_path):
            logging.info(f"Используется существующий файла {file_path}")
            df = pd.read_csv(file_path)    
            logging.info(f"Всего записей: {len(df)}.")
            return df
              
    logging.info(f"Начало получения данных для символа {symbol}")
    # Создание директории, если её нет
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Определение интервала в миллисекундах
    interval_ms = {
        "1m": 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
    }.get(
        timeframe, 5 * 60 * 1000
    )  # По умолчанию 5 минут

    # Проверка существования файла и загрузка существующих данных
    if os.path.exists(file_path):
        logging.info(f"Файл {file_path} найден. Загрузка существующих данных.")
        df = pd.read_csv(file_path)

        # Преобразуем столбец timestamp: строки в datetime, числа остаются как есть
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # Для строковых дат конвертируем их в миллисекунды
        df["timestamp"] = df["timestamp"].apply(
            lambda x: int(x.timestamp() * 1000) if not pd.isnull(x) else np.nan
        )

        # Убедимся, что нет пропущенных значений
        df = df.dropna(subset=["timestamp"])

        # Преобразуем timestamp в целочисленный тип
        df["timestamp"] = df["timestamp"].astype(np.int64)

        # Получаем временную метку последней записи
        since = df["timestamp"].iloc[-1] + 1

        # Проверка, нужно ли увеличить limit, чтобы покрыть весь пропущенный интервал
        now = exchange.milliseconds()
        logging.info(
            f"Текущее время на бирже: {pd.to_datetime(now, unit='ms', utc=True)}"
        )
        since = now - (limit * interval_ms)
        missing_data_points = (now - since) // interval_ms
        if missing_data_points > limit:
            logging.info(
                f"Пропущенный интервал превышает limit ({limit}). Увеличение limit до {missing_data_points}."
            )
            limit = missing_data_points

        all_ohlcv = df[
            ["timestamp", "open", "high", "low", "close", "volume"]
        ].values.tolist()
    else:
        logging.info(f"Файл {file_path} не найден. Будет создан новый файл.")
        # Если файл не существует, вычисляем since на основе limit
        now = exchange.milliseconds()
        logging.info(
            f"Текущее время на бирже: {pd.to_datetime(now, unit='ms', utc=True)}"
        )
        since = now - (limit * interval_ms)

    # Получение новых данных порциями по 900 записей
    while True:
        try:
            fetch_limit = 900  # min(1000, limit - len(all_ohlcv))
            ohlcv = await exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since, limit=fetch_limit
            )
            ohlcv = await exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since, limit=fetch_limit
            )

            if not ohlcv:
                logging.debug("Нет новых данных для загрузки")
                break

            all_ohlcv.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]

            # Обновляем since для следующего запроса
            since = last_timestamp + interval_ms

            # Проверяем, достигли ли текущего момента времени
            if last_timestamp + interval_ms >= exchange.milliseconds():
                logging.debug("Достигнут конец доступных данных на бирже")
                break

        except Exception as e:
            logging.error(f"Ошибка при получении данных: {e}\n", exc_info=True)
            break

    # Создание DataFrame и сохранение данных в файл
    df = pd.DataFrame(
        all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.drop_duplicates(subset="timestamp", keep="last", inplace=True)
    df.sort_values(by="timestamp", inplace=True)

    df.to_csv(file_path, index=False)
    logging.info(f"Данные сохранены в файл {file_path}. Всего записей: {len(df)}")

    # Возвращаем последние 'limit' записей
    # print(return_limit)
    # await is_continue(exchange)
    return df.tail(return_limit)


async def list_available_symbols(exchange):
    try:
        await exchange.load_markets()
        logging.debug("Рынки загружены")
        return exchange.symbols
    except Exception as e:
        logging.error(f"Ошибка при загрузке рынков: {e}\n", exc_info=True)
        return []


async def verify_symbol(exchange, symbol, run_mode=None):
    if run_mode == "BK":
        return True
    try:
        await exchange.load_markets()
        is_valid = symbol in exchange.symbols
        logging.debug(
            f"Проверка символа {symbol}: {'доступен' if is_valid else 'недоступен'}"
        )
        return is_valid
    except Exception as e:
        logging.error(f"Ошибка при проверке символа: {e}\n", exc_info=True)
        return False


def get_real_balance_sync(exchange, run_mode=None):
    if run_mode != "TRADE":
        return 10
    try:
        balance = asyncio.run(exchange.fetch_balance())
        real_balance = balance["total"].get("USDT", 0)
        logging.debug(f"Текущий баланс: {real_balance} USDT")
        return real_balance
    except Exception as e:
        logging.error(f"Ошибка при получении баланса: {e}\n", exc_info=True)
        return None


async def get_real_balance_async(exchange, run_mode=None):
    if run_mode != "TRADE":
        return 10
    try:
        balance = await exchange.fetch_balance()
        real_balance = balance["total"].get("USDT", 0)
        logging.debug(f"Текущий баланс (асинхронно): {real_balance} USDT")
        return real_balance
    except Exception as e:
        logging.error(f"Ошибка при получении баланса: {e}\n", exc_info=True)
        return None

def shutdown_handler():
    logging.info("Обработка сигнала завершения")
    for task in asyncio.all_tasks():
        task.cancel()


def clear_log_file(filename):
    """
    Удаляет или очищает файл логов перед началом работы.
    :param filename: Имя файла логов.
    """
    try:
        # Проверяем, существует ли файл
        if os.path.exists(filename):
            # Очищаем содержимое файла
            with open(filename, "w", encoding="utf-8"):
                pass
        # Если файл не существует, он будет создан при записи логов
    except Exception as e:
        print(f"Ошибка при очистке файла логов: {e}\n", exc_info=True)


def create_date_mask(df, start_date, end_date):
    # Логика фильтрации для create_result_df
    if start_date and end_date:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        # Фильтрация по диапазону дат
        mask = (df["timestamp"].dt.date >= pd.to_datetime(start_date).date()) & (
            df["timestamp"].dt.date <= pd.to_datetime(end_date).date()
        )
    elif start_date:
        # Фильтрация от start_date до конца данных
        mask = df["timestamp"].dt.date >= pd.to_datetime(start_date).date()
    elif end_date:
        # Фильтрация от начала данных до end_date
        mask = df["timestamp"].dt.date <= pd.to_datetime(end_date).date()
    else:
        # Если оба параметра отсутствуют, использовать все данные
        mask = pd.Series([True] * len(df))
    return mask

def create_result_df(df, stat_dir):
    if df.empty:
        logging.error("DF is empty")
        return
    
    def calculate_diff(row):
        if row["type"] == "long":
            return (
                (row["close_price"] - row["open_price"]) / row["open_price"]
            ) * 100
        elif row["type"] == "short":
            return (
                (row["open_price"] - row["close_price"]) / row["open_price"]
            ) * 100
        return 0

    # Применение функции для создания нового столбца
    df["diff_PCT"] = df.apply(calculate_diff, axis=1)
    
    df.dropna(inplace=True)
    if len(df)  > 1 :
        df.to_csv(f"{stat_dir}/result.csv")
    else:
        logging.error("DF is empty, no data to save")
        return


    # Рсчет статистики
    # df = df[: len(df)-1]
    # Метрики для расчёта
    # 1. Общее количество diff_PCT > 0.2 и процентное отношение к количеству сделок
    total_diff_pct_gt_0_2 = df[df["diff_PCT"] > 0.2].shape[0]
    # Всего сделок

    total_trades = len(df)
    
    # Конечный баланс
    end_balance = df["balance"].iloc[-1]
    
    if np.isnan(end_balance):
        end_balance = df["balance"].iloc[-2]
        total_trades = total_trades - 1
    
    diff_pct_gt_0_2_ratio = (
        (total_diff_pct_gt_0_2 / total_trades * 100) if total_trades > 0 else 0
    )

    # 2. Общее количество diff_PCT > 0.2 при type=="long" и процентное отношение
    long_trades = df[df["type"] == "long"]
    long_diff_pct_gt_0_2 = long_trades[long_trades["diff_PCT"] > 0.2].shape[0]
    long_diff_pct_gt_0_2_ratio = (
        (long_diff_pct_gt_0_2 / long_trades.shape[0] * 100)
        if long_trades.shape[0] > 0
        else 0
    )

    # 3. Общее количество положительных profit при type=="short" и процентное отношение
    short_trades = df[df["type"] == "short"]
    short_positive_profit = short_trades[short_trades["profit"] > 0].shape[0]
    short_positive_profit_ratio = (
        (short_positive_profit / short_trades.shape[0] * 100)
        if short_trades.shape[0] > 0
        else 0
    )

    # 4. Общее количество положительных profit и процентное отношение к количеству сделок
    total_positive_profit = df[df["profit"] > 0].shape[0]
    total_positive_profit_ratio = (
        (total_positive_profit / total_trades * 100) if total_trades > 0 else 0
    )
    total_positive_profit_ratio = round(total_positive_profit_ratio,2)

    # 5. Общее количество положительных profit при type=="long" и процентное отношение
    long_positive_profit = long_trades[long_trades["profit"] > 0].shape[0]
    long_positive_profit_ratio = (
        (long_positive_profit / long_trades.shape[0] * 100)
        if long_trades.shape[0] > 0
        else 0
    )
    long_positive_profit_ratio = round(long_positive_profit_ratio,2)

    # 6. Общее количество положительных profit при type=="short" и процентное отношение
    short_positive_profit = short_trades[short_trades["profit"] > 0].shape[0]
    short_positive_profit_ratio = (
        (short_positive_profit / short_trades.shape[0] * 100)
        if short_trades.shape[0] > 0
        else 0
    )
    short_positive_profit_ratio = round(short_positive_profit_ratio,2)

    # Дополнительные расчёты
    # Самая прибыльная и убыточная сделка во всём DataFrame
    max_profit_trade = df["profit"].max() if not df.empty else None
    min_profit_trade = df["profit"].min() if not df.empty else None

    # Самая прибыльная и убыточная сделка для long
    max_profit_long = long_trades["profit"].max() if not long_trades.empty else None
    min_profit_long = long_trades["profit"].min() if not long_trades.empty else None

    # Самая прибыльная и убыточная сделка для short
    max_profit_short = short_trades["profit"].max() if not short_trades.empty else None
    min_profit_short = short_trades["profit"].min() if not short_trades.empty else None

    # 7. Общее количество сделок
    # total_trades = len(df)

    begin_ts = df["open_ts"].iloc[1]
    end_ts = df["close_ts"].iloc[-1]
    if end_ts != end_ts:
       end_ts = df["close_ts"].iloc[-2] 

    # Дополнительный расчет  статистики
    # Фильтруем сделки по типам
    long_trades = df[df["type"] == "long"]
    short_trades = df[df["type"] == "short"]

    #Продолжительность сделок
    max_duration_long = long_trades["duration"].max()
    mean_duration_long = long_trades["duration"].mean()
    max_duration_short = short_trades["duration"].max()
    mean_duration_lshort = short_trades["duration"].mean()
    

    # Расчет для сделок long
    total_long_profit = long_trades[long_trades["profit"] > 0]["profit"].sum()
    total_long_loss = long_trades[long_trades["profit"] < 0]["profit"].sum()
    long_profit_loss_ratio = (
        (total_long_profit / abs(total_long_loss))
        if total_long_loss != 0
        else float("inf")
    )

    # Расчет для сделок short
    total_short_profit = short_trades[short_trades["profit"] > 0]["profit"].sum()
    total_short_loss = short_trades[short_trades["profit"] < 0]["profit"].sum()
    short_profit_loss_ratio = (
        (total_short_profit / abs(total_short_loss))
        if total_short_loss != 0
        else float("inf")
    )

    # Разницы между прибыльными и убыточными сделками
    long_profit_loss_diff = total_long_profit + total_long_loss
    short_profit_loss_diff = total_short_profit + total_short_loss
    
    # Прибыль в %
    end_profit_ratio = (end_balance - 10)*10

    # Итоговый DataFrame с результатами
    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Start Date: ",
                "Stop Date:  ",
                # "Total diff_PCT > 0.2",
                "Ratio positive profit to all trades (%): ",
                "Ratio positive profit to long trades (%): ",
                "Ratio positive profit to short trades (%): ",
                "Total positive trades: ",
                "Total positive trades (long):  ",
                "Total positive trades (short): ",
                
                "Total profit (long): ",
                "Total loss  (long): ",
                "Ratio profit/loss (long): ",
                "PL (long): ",
                
                "Total profit (short): ",
                "Total loss  (short): ",
                "Ratio profit/loss (short): ",             
                "PL (short): ",
                
            
                "Ratio diff_PCT > 0.2 to all trades (%): ",
                "Max profit trade: ",
                "Max loss trade: ",
                "Max profit trade (long): ",
                "Max loss trade (long): ",
                "Max profit trade (short): ",
                "Max loss trade (short): ",
                "Max duration (long): ",
                "Mean duration (long): ",
                "Max duration  (short): ",
                "Mean duration (short): ",
                "Balance: ",
                "Total Profit (%): ",
                "Total Trades: ",
                # "Total diff_PCT > 0.2 (long)",
                # "Ratio diff_PCT > 0.2 to long trades (%)",
            ],
            "Value": [
                begin_ts,
                end_ts,
                # total_diff_pct_gt_0_2,
                total_positive_profit_ratio,
                long_positive_profit_ratio,
                short_positive_profit_ratio,
                total_positive_profit,
                long_positive_profit,
                short_positive_profit,
                
                total_long_profit,
                total_long_loss,
                round(long_profit_loss_ratio, 2),
                long_profit_loss_diff,
                
                total_short_profit,
                total_short_loss,
                round(short_profit_loss_ratio, 2), 
                short_profit_loss_diff,

                round(diff_pct_gt_0_2_ratio, 2),
                max_profit_trade,
                min_profit_trade,
                max_profit_long,
                min_profit_long,
                max_profit_short,
                min_profit_short,
                max_duration_long,
                round(mean_duration_long, 2),
                max_duration_short,
                round(mean_duration_lshort, 2),
                end_balance,
                round(end_profit_ratio, 2),
                total_trades,
                # long_diff_pct_gt_0_2,
                # long_diff_pct_gt_0_2_ratio,
            ],
        }
    )

    # Сохранение в файл CSV
    output_path = f"{stat_dir}/stat.csv"
    if not summary_df.empty:
        summary_df.to_csv(output_path, index=False)
        print(f"Summary saved to {output_path}")
    else:
        logging.error("DF is empty, No data to save")
    return