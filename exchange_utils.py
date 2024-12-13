# Utilities for working with a exchange and others
import sys
import os
import asyncio
import logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv


load_dotenv()


DATA_DIR = os.getenv("DATA_DIR")


async def is_continue(exchange, exit=False):
    """For correct exit. Debug only"""
    if exit:
        user_choice = "Y"
    else:
        user_choice = input("Продолжить ? (Y/n): ")
    if user_choice in ["n", " N", "т", "Т"]:
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
async def get_full_data(exchange, symbol, timeframe="5m", since=None, limit=2016):
    return_limit = limit  # Количество возврашаемых свечей
    all_ohlcv = []
    logging.info(f"Начало получения данных для символа {symbol}")

    file_path = create_file_path(symbol, timeframe)

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
            logging.error(f"Ошибка при получении данных: {e}")
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
    print(return_limit)
    # await is_continue(exchange)
    return df.tail(return_limit)


async def list_available_symbols(exchange):
    try:
        await exchange.load_markets()
        logging.debug("Рынки загружены")
        return exchange.symbols
    except Exception as e:
        logging.error(f"Ошибка при загрузке рынков: {e}")
        return []


async def verify_symbol(exchange, symbol):
    try:
        await exchange.load_markets()
        is_valid = symbol in exchange.symbols
        logging.debug(
            f"Проверка символа {symbol}: {'доступен' if is_valid else 'недоступен'}"
        )
        return is_valid
    except Exception as e:
        logging.error(f"Ошибка при проверке символа: {e}")
        return False


def get_real_balance_sync(exchange):
    try:
        balance = asyncio.run(exchange.fetch_balance())
        real_balance = balance["total"].get("USDT", 0)
        logging.debug(f"Текущий баланс: {real_balance} USDT")
        return real_balance
    except Exception as e:
        logging.error(f"Ошибка при получении баланса: {e}")
        return None


async def get_real_balance_async(exchange):
    try:
        balance = await exchange.fetch_balance()
        real_balance = balance["total"].get("USDT", 0)
        logging.debug(f"Текущий баланс (асинхронно): {real_balance} USDT")
        return real_balance
    except Exception as e:
        logging.error(f"Ошибка при получении баланса: {e}")
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
        print(f"Ошибка при очистке файла логов: {e}")


def create_date_mask(df, start_date, end_date):
    # Логика фильтрации
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
