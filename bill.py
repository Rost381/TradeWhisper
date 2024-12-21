import pandas as pd
import numpy as np
from ta.momentum import AwesomeOscillatorIndicator
from ta.trend import SMAIndicator
from ta.volatility import BollingerBands

# === Загрузка данных ===
data = pd.read_csv("futures_data_5m/DOGEUSDT.csv")  # Замените на ваш файл данных
data = data[:8740]
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)

# Преобразование типов
numeric_columns = ['high', 'low', 'close', 'volume']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Удаление строк с некорректными данными
data.dropna(subset=numeric_columns, inplace=True)

# === Инициализация индикаторов ===
# Alligator (челюсти, зубы, губы)
data['alligator_jaw'] = SMAIndicator(data['close'], window=13).sma_indicator()
data['alligator_teeth'] = SMAIndicator(data['close'], window=8).sma_indicator()
data['alligator_lips'] = SMAIndicator(data['close'], window=5).sma_indicator()

# Awesome Oscillator (AO)
ao = AwesomeOscillatorIndicator(data['high'], data['low'])
data['ao'] = ao.awesome_oscillator()

# Accelerator Oscillator (AC)
data['ac'] = data['ao'] - SMAIndicator(data['ao'], window=5).sma_indicator()

# Gator Oscillator
data['gator_upper'] = abs(data['alligator_lips'] - data['alligator_teeth'])
data['gator_lower'] = abs(data['alligator_teeth'] - data['alligator_jaw'])

# Market Facilitation Index (MFI)
data['mfi'] = (data['high'] - data['low']) / data['volume']

# Bollinger Bands
bollinger = BollingerBands(close=data['close'], window=20, window_dev=2)
data['bb_upper'] = bollinger.bollinger_hband()
data['bb_lower'] = bollinger.bollinger_lband()
data['bb_middle'] = bollinger.bollinger_mavg()

# === Логика фракталов ===
def calculate_fractals(df):
    df['fractal_high'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
    df['fractal_low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
    return df

data = calculate_fractals(data)

# === Торговая логика ===
def generate_signals(df):
    df['signal'] = 0  # 0 = нет позиции, 1 = покупка, -1 = продажа

    for i in range(2, len(df)):
        # Проверка на тренд Alligator (губы > зубы > челюсти)
        is_uptrend = (
            df['alligator_lips'].iloc[i] > df['alligator_teeth'].iloc[i] > df['alligator_jaw'].iloc[i]
        )
        is_downtrend = (
            df['alligator_lips'].iloc[i] < df['alligator_teeth'].iloc[i] < df['alligator_jaw'].iloc[i]
        )

        # Дополнительные подтверждения
        ac_positive = df['ac'].iloc[i] > 0
        ac_negative = df['ac'].iloc[i] < 0
        mfi_high = df['mfi'].iloc[i] > df['mfi'].mean()
        gator_expanding = (
            df['gator_upper'].iloc[i] > df['gator_upper'].iloc[i - 1] and
            df['gator_lower'].iloc[i] > df['gator_lower'].iloc[i - 1]
        )

        # Учет Bollinger Bands
        price_above_bb_upper = df['close'].iloc[i] > df['bb_upper'].iloc[i]
        price_below_bb_lower = df['close'].iloc[i] < df['bb_lower'].iloc[i]
        price_near_bb_middle = abs(df['close'].iloc[i] - df['bb_middle'].iloc[i]) / df['bb_middle'].iloc[i] < 0.01

        # Логика открытия позиций
        if is_uptrend and df['ao'].iloc[i] > 0 and ac_positive and mfi_high and gator_expanding and price_near_bb_middle and df['fractal_high'].iloc[i - 1]:
            df['signal'].iloc[i] = 1  # Сигнал на покупку
        elif is_downtrend and df['ao'].iloc[i] < 0 and ac_negative and not mfi_high and gator_expanding and price_below_bb_lower and df['fractal_low'].iloc[i - 1]:
            df['signal'].iloc[i] = -1  # Сигнал на продажу

    return df

# Генерация сигналов
data = generate_signals(data)

# === Симуляция стратегии ===
def backtest_strategy(df, initial_balance=10, risk_per_trade=1):
    balance = initial_balance
    position = 0
    trades = []

    for i in range(len(df)):
        if df['signal'].iloc[i] == 1 and position == 0:  # Покупка
            open_ts = df.index[i]
            open_price = df['close'].iloc[i]
            position = balance * risk_per_trade / open_price
            balance -= position * open_price
        elif df['signal'].iloc[i] == -1 and position > 0:  # Закрытие позиции
            close_ts = df.index[i]
            close_price = df['close'].iloc[i]
            profit = position * (close_price - open_price)
            balance += position * close_price
            duration = (close_ts - open_ts).total_seconds() / 60

            trades.append({
                'type': 'long',
                'open_ts': open_ts,
                'open_price': open_price,
                'close_ts': close_ts,
                'close_price': close_price,
                'duration': duration,
                'profit': profit,
                'balance': balance
            })

            position = 0

    # Закрытие открытых позиций
    if position > 0:
        close_ts = df.index[-1]
        close_price = df['close'].iloc[-1]
        profit = position * (close_price - open_price)
        balance += position * close_price
        duration = (close_ts - open_ts).total_seconds() / 60

        trades.append({
            'type': 'long',
            'open_ts': open_ts,
            'open_price': open_price,
            'close_ts': close_ts,
            'close_price': close_price,
            'duration': duration,
            'profit': profit,
            'balance': balance
        })

    # Сохранение результатов в файл
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv('stats_bill/result.csv', index_label='index')

    return balance

# Запуск симуляции
final_balance = backtest_strategy(data)
print(f"Итоговый баланс: {final_balance:.2f}")
