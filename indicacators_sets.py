import logging
import numpy as np
from ta import trend, volatility, momentum
from ta.momentum import RSIIndicator, StochasticOscillator, UltimateOscillator
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
from ta.volatility import AverageTrueRange
from ta.trend import IchimokuIndicator, PSARIndicator, CCIIndicator, TRIXIndicator, MACD
from scipy.signal import find_peaks

def ema_angle(df,period=50):
    # Расчет угла наклона EMA50
    df["ema_long"] = trend.EMAIndicator(df["close"], window=period).ema_indicator()  # Длинная EMA
    df["ema_angle"] = np.arctan(df["ema_long"].diff())  # Угол в градусах
    return df

def find_sr(df, dist=48):
    # Найдем пики (локальные максимумы)
    peaks, _ = find_peaks(df["high"], distance=dist)
    # Найдем впадины (локальные минимумы)
    troughs, _ = find_peaks(-df["low"], distance=dist)
    support_levels = df["low"].iloc[troughs]
    resistance_levels = df["high"].iloc[peaks]
    return peaks, troughs, support_levels, resistance_levels


def find_sr_lines(df, dist=48):
    # Найдем пики (локальные максимумы)
    peaks, _ = find_peaks(df["high"], distance=dist)
    # Найдем впадины (локальные минимумы)
    troughs, _ = find_peaks(-df["low"], distance=dist)
    support_levels = df["low"].iloc[troughs]
    resistance_levels = df["high"].iloc[peaks]
    df["support"] = support_levels
    df["resistance"] = resistance_levels
    # df[["support", "resistance"]] = df[["support", "resistance"]].fillna(method="ffill")
    df[["support", "resistance"]] = df[["support", "resistance"]].ffill()
    return df


# Функция для расчёта Pivot Points для заданного окна
def calculate_pivot_points_dynamic(df, window):
    rolling = df.rolling(window=window)
    df["pivot"] = (
        rolling["high"].max() + rolling["low"].min() + rolling["close"].mean()
    ) / 3
    df["r1"] = 2 * df["pivot"] - rolling["low"].min()
    df["s1"] = 2 * df["pivot"] - rolling["high"].max()
    df["r2"] = df["pivot"] + (rolling["high"].max() - rolling["low"].min())
    df["s2"] = df["pivot"] - (rolling["high"].max() - rolling["low"].min())
    df["r3"] = df["r2"] + (rolling["high"].max() - rolling["low"].min())
    df["s3"] = df["s2"] - (rolling["high"].max() - rolling["low"].min())
    return df


# Функция для расчёта фракталов для заданного окна
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

def calculate_rvi(df, window=10):
    close_open = df["close"] - df["open"]
    high_low = df["high"] - df["low"]
    rvi = close_open / high_low
    rvi = rvi.rolling(window=window).mean()
    logging.debug("RVI рассчитан")
    return rvi

# ------------------
def add_technical_indicators_v0(df):
    df = df.copy()
    logging.debug("Добавление технических индикаторов")
    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)
    logging.debug("Технические индикаторы добавлены")
    return df

# ------------------
def add_technical_indicators_original(df):
    logging.debug("Добавление технических индикаторов")
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['ema20'] = trend.EMAIndicator(df['close'], window=20).ema_indicator()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    bollinger = volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bollinger_hband'] = bollinger.bollinger_hband()
    df['bollinger_lband'] = bollinger.bollinger_lband()
    df['stoch'] = StochasticOscillator(df['high'], df['low'], df['close'], window=14).stoch()
    cumulative_volume = df['volume'].cumsum()
    cumulative_volume[cumulative_volume == 0] = 1
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / cumulative_volume
    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    ichimoku = IchimokuIndicator(df['high'], df['low'], window1=9, window2=26, window3=52)
    df['ichimoku_a'] = ichimoku.ichimoku_a()
    df['ichimoku_b'] = ichimoku.ichimoku_b()
    df['psar'] = PSARIndicator(df['high'], df['low'], df['close'], step=0.02, max_step=0.2).psar()
    df['cci'] = CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
    df['trix'] = TRIXIndicator(df['close'], window=15).trix()
    df['ultimate_osc'] = UltimateOscillator(df['high'], df['low'], df['close'], window1=7, window2=14, window3=28).ultimate_oscillator()
    df['rvi'] = calculate_rvi(df, window=10)
    df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df['chaikin_money_flow'] = ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'], window=20).chaikin_money_flow()
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)
    logging.debug("Технические индикаторы добавлены")
    return df
# ------------------

def add_technical_indicators_v2(df):
    logging.debug("Добавление технических индикаторов")
    df = df.copy()

    df['rvi'] = calculate_rvi(df, window=10)
    # Индикаторы тренда
    df["ema7"] = trend.EMAIndicator(df["close"], window=7).ema_indicator()  # Короткая EMA
    df["ema21"] = trend.EMAIndicator(df["close"], window=21).ema_indicator()  # Средняя EMA
    df["ema50"] = trend.EMAIndicator(df["close"], window=50).ema_indicator()  # Длинная EMA
    df['adx'] = trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()  # Индикатор силы тренда

    # Индикаторы осцилляторы
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()  # Индекс относительной силы
    df['stoch'] = StochasticOscillator(df['high'], df['low'], df['close'], window=9).stoch()  # Стохастик
    df['ultimate_osc'] = UltimateOscillator(df['high'], df['low'], df['close'], window1=3, window2=7, window3=14).ultimate_oscillator()

    # Индикаторы волатильности
    df['bollinger_hband'] = volatility.BollingerBands(df['close'], window=20, window_dev=2).bollinger_hband()
    df['bollinger_lband'] = volatility.BollingerBands(df['close'], window=20, window_dev=2).bollinger_lband()
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['donchian_upper'] = volatility.DonchianChannel(df["close"], df['high'], df['low'], window=20).donchian_channel_hband()
    df["donchian_lower"] = volatility.DonchianChannel(df["close"], df["high"], df["low"], window=20).donchian_channel_lband()
    
    # Индикаторы объема
    df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()  # Баланс объема
    df['cmf'] = ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'], window=20).chaikin_money_flow()  # Chaikin Money Flow
    cumulative_volume = df['volume'].cumsum()
    cumulative_volume[cumulative_volume == 0] = 1
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / cumulative_volume

    # Индикаторы прочие
    df['cci'] = CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()  # Индекс товарного канала

    # Добавление уровней
    df = find_sr_lines(df, dist=48)
    
    # Заполнение пропущенных значений

    logging.debug("Технические индикаторы добавлены")

 
    df.dropna(inplace=True)
    # df.bfill(inplace=True)
    # df.fillna(0, inplace=True)

    # logging.debug("Данные старшего таймфрейма добавлены")

    return df
#-------------------------------------------------------------------
# v2 Bill

def add_technical_indicators_v3(df):
    logging.debug("Добавление технических индикаторов")
    df = df.copy()
    
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    # Alligator (челюсти, зубы, губы)
    df['alligator_jaw'] = trend.SMAIndicator(df['close'], window=13).sma_indicator()
    df['alligator_teeth'] = trend.SMAIndicator(df['close'], window=8).sma_indicator()
    df['alligator_lips'] = trend.SMAIndicator(df['close'], window=5).sma_indicator()

    # Awesome Oscillator (AO)
    ao = momentum.AwesomeOscillatorIndicator(df['high'], df['low'])
    df['ao'] = ao.awesome_oscillator()

    # Accelerator Oscillator (AC)
    df['ac'] = df['ao'] - trend.SMAIndicator(df['ao'], window=5).sma_indicator()

    # Gator Oscillator
    df['gator_upper'] = abs(df['alligator_lips'] - df['alligator_teeth'])
    df['gator_lower'] = abs(df['alligator_teeth'] - df['alligator_jaw'])

    # Market Facilitation Index (MFI)
    df['mfi'] = (df['high'] - df['low']) / df['volume']

    # Bollinger Bands
    bollinger = volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_upper'] = bollinger.bollinger_hband()
    df['bb_lower'] = bollinger.bollinger_lband()
    df['bb_middle'] = bollinger.bollinger_mavg()

# === Логика фракталов ===
    df = calculate_fractals(df, window=5)  # 5 свечей
    # df[["fractal_high", "fractal_low"]] = df[["fractal_high", "fractal_low"]].fillna(method="ffill")
    df[["fractal_high", "fractal_low"]] = df[["fractal_high", "fractal_low"]].ffill()

    df.dropna(inplace=True)

    return df