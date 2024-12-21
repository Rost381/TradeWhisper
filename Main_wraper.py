import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys
import asyncio
import logging
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import ccxt.async_support as ccxt_async
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure


from dotenv import load_dotenv
from collections import deque
import signal
import copy
import json
import optuna
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


from tw_utils import *
from indicacators_sets import *



# Set global vars
load_dotenv()

RUN_MODE = os.getenv("RUN_MODE")   
SYMBOL = os.getenv("SYMBOL")
TIMEFRAME = os.getenv("TIMEFRAME")
ACC_MODE = os.getenv("ACC_MODE")
DATA_DIR = os.getenv("DATA_DIR")
MODELS_DIR = os.getenv("MODELS_DIR")
STATS_DIR = os.getenv("STATS_DIR")
MODEL_SUFFIX = os.getenv("MODEL_SUFFIX")

STATS_DIR = f"{STATS_DIR}_{MODEL_SUFFIX}"

if RUN_MODE=="BK":
   load_dotenv(".env-bk", override=True)     
BK_DIR = os.getenv("BK_DIR", None)
BK_START = os.getenv("BK_START")
BK_END = os.getenv("BK_END")    
LOGS_TO_FILE = os.getenv("LOGS_TO_FILE")
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
log_filename = f"{script_name}.log"

if LOGS_TO_FILE == "DISABLE":   
    handlers = [
        logging.StreamHandler(sys.stdout),  # Вывод в консоль
    ]
else:
    handlers = [
        logging.StreamHandler(sys.stdout),  # Вывод в консоль
        logging.FileHandler(log_filename, mode="a", encoding="utf-8"),  # Вывод в файл
    ]

logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
    handlers=handlers,
)

if ACC_MODE == "LIVE":  
    API_KEY = os.getenv("API_KEY", None)
    API_SECRET = os.getenv("API_SECRET", None)
else:
    API_KEY = os.getenv("API_KEY_DEMO", None)
    API_SECRET = os.getenv("API_SECRET_DEMO", None)

exchange_config = {
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'adjustForTimeDifference': True,
        'recvWindow': 10000
    },
    'timeout': 30000
}

if MODEL_SUFFIX[:2] == "L0":
    GET_DATA_LIMIT = 2000
    MODEL_HISTORY_SIZE=100
    MODEL_WINDOWS_SIZE=20 
elif MODEL_SUFFIX[:2] == "L1":
    GET_DATA_LIMIT = 4320   
    MODEL_HISTORY_SIZE=240
    MODEL_WINDOWS_SIZE=48 
else:
    logging.error("Need suffix")
    sys.exit(1)

if not os.path.exists(STATS_DIR):
    os.makedirs(STATS_DIR)
    
with open(f"{STATS_DIR}/model_param.json", 'w') as f:
    model_param ={
        "SYMBOL" : SYMBOL,
        "TIMEFRAME" : TIMEFRAME,
        "GET_DATA_LIMIT": GET_DATA_LIMIT,
        "MODEL_HISTORY_SIZE" : MODEL_HISTORY_SIZE,
        "MODEL_WINDOWS_SIZE" : MODEL_WINDOWS_SIZE
        
    }  
    json.dump(model_param, f)

class TradingEnvironment(gym.Env):
    def __init__(self, data, norm_params=None, is_backtest=False, initial_balance=10, risk_percentage=0.7, 
                 short_term_threshold=10, long_term_threshold=50, history_size=MODEL_HISTORY_SIZE, window_size=MODEL_WINDOWS_SIZE):
        super(TradingEnvironment, self).__init__()
        logging.debug("Initializing TradingEnvironment")
        self.is_backtest = is_backtest
        self.timestamps = data['timestamp'].reset_index(drop=True)
        self.data = data.drop(columns=['timestamp']).reset_index(drop=True)
        self.initial_balance = initial_balance
        self.risk_percentage = risk_percentage
        self.short_term_threshold = short_term_threshold
        self.long_term_threshold = long_term_threshold
        self.window_size = window_size
        if norm_params is None:
            self.means = self.data.mean()
            self.stds = self.data.std().replace(0, 1e-8)
        else:
            self.means = pd.Series(norm_params['means'])
            self.stds = pd.Series(norm_params['stds'])
        self.normalized_data = (self.data - self.means) / self.stds
        low = self.normalized_data.min().values - 1
        high = self.normalized_data.max().values + 1
        num_features = self.data.shape[1]
        self.observation_space = spaces.Box(
            low=np.tile(low, self.window_size).astype(np.float32),
            high=np.tile(high, self.window_size).astype(np.float32),
            shape=(self.window_size * num_features,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # Изменено с 4 на 3
        self.obs_window = deque(maxlen=self.window_size)
        self.history = deque(maxlen=history_size)
        self.reset()
        self.save_state()

        # Create dataframe for stat
        if self.is_backtest:
            self.result_df = pd.DataFrame(
            columns=[
                "type",
                "open_ts",
                "open_price",
                "close_ts",
                "close_price",
                "duration",
                "profit",
                "diff_PCT",
                "balance",
            ]
        )
    def reset(self, *, seed=None, options=None):
        logging.debug("Resetting environment")
        self.balance = self.initial_balance
        self.previous_balance = self.initial_balance
        self.position = None
        self.entry_price = 0
        self.entry_step = 0
        self.current_step = 0
        self.done = False
        self.total_profit = 0
        self.positions = []
        self.position_size = 0
        self.units = 0
        self.balance_history = [self.balance]
        self.history.clear()
        self.obs_window.clear()
        initial_window = self.normalized_data.iloc[self.current_step:self.current_step + self.window_size]
        for _, row in initial_window.iterrows():
            self.obs_window.append(row.values.astype(np.float32))
        self.current_step += self.window_size
        self.save_state()
        return self._get_observation(), {}

    def _get_observation(self):
        if len(self.obs_window) < self.window_size:
            padding = [np.zeros(self.normalized_data.shape[1], dtype=np.float32)] * (self.window_size - len(self.obs_window))
            window = list(padding) + list(self.obs_window)
        else:
            window = list(self.obs_window)
        obs = np.concatenate(window)
        return obs.astype(np.float32)

    def save_state(self):
        state = {
            'balance': self.balance,
            'position': self.position,
            'entry_price': self.entry_price,
            'entry_step': self.entry_step,
            'current_step': self.current_step,
            'done': self.done,
            'total_profit': self.total_profit,
            'positions': copy.deepcopy(self.positions),
            'position_size': self.position_size,
            'units': self.units,
            'balance_history': copy.deepcopy(self.balance_history),
            'obs_window': copy.deepcopy(self.obs_window),
            'previous_balance': self.previous_balance
        }
        self.history.append(state)

    def load_state(self, steps_back=2):
        if len(self.history) >= steps_back:
            state = self.history[-steps_back]
            self.balance = state['balance']
            self.position = state['position']
            self.entry_price = state['entry_price']
            self.entry_step = state['entry_step']
            self.current_step = state['current_step']
            self.done = state['done']
            self.total_profit = state['total_profit']
            self.positions = copy.deepcopy(state['positions'])
            self.position_size = state['position_size']
            self.units = state['units']
            self.balance_history = copy.deepcopy(state['balance_history'])
            self.obs_window = copy.deepcopy(state['obs_window'])
            self.previous_balance = state['previous_balance']
            logging.debug("State loaded successfully")
        else:
            logging.warning("Недостаточно истории для отката")

    def detect_error(self):
        if self.balance < self.initial_balance * 0.5:
            logging.error("Баланс упал ниже 50%")
            # Пишем результат в df
            if self.is_backtest:
                create_result_df(self.result_df, STATS_DIR)
                sys.exit[1]
            return True
        return False

    def handle_error(self):
        logging.info("Обработка ошибки путем отката состояния")
        self.load_state(steps_back=2)

    def step(self, action):
        self.save_state()
        reward = 0
        info = {}
        if self.current_step >= len(self.data):
            self.done = True
            profit = self.balance - self.previous_balance
            volatility = self.data['atr'].iloc[self.current_step - 1]
            reward = profit / (volatility + 1e-8)
            logging.debug(
                f"Эпизод завершен. Прибыль: {profit}, Волатильность: {volatility}, Награда: {reward}, Время: {timestamp}")
            return self._get_observation(), reward, self.done, False, info
        price = self.data['close'].iloc[self.current_step]
        timestamp = self.timestamps[self.current_step]
        atr = self.data['atr'].iloc[self.current_step]
        logging.debug(f"Текущий шаг: {self.current_step}, Цена: {price}, Время: {timestamp}, ATR: {atr}")

        if action == 0:
            logging.debug("Действие: Удерживать позицию / Ждать")
            pass

        elif action == 1:
            # if self.position == 'short':
            #     logging.info(
            #         f"Действие: Переключение с шорт на лонг, Время: {timestamp}")
            #     reward += self._close_position(price, timestamp)
            if self.position != 'long':
                logging.info(f"Действие:  Открыть длинную позицию, Время: {timestamp}")
                self._open_position('long', price, timestamp, atr)

        elif action == 2:
            if self.position == 'long':
                # logging.info(
                #     f"Действие: Переключение с лонг на шорт, Время: {timestamp},")
                logging.info(
                    f"Действие: Закрыть лонг, Время: {timestamp},")
                reward += self._close_position(price, timestamp)
            # if self.position != 'short':
            #     logging.info(f"Действие: Открыть короткую позицию, Время: {timestamp}")
            #     self._open_position('short', price, timestamp, atr)
            if self.position is None:
                reward = -0.01 # Штраф за закрытие неоткрытой позиции

        # Удаляем логику тейк-профита и стоп-лосса

        profit = self.balance - self.previous_balance
        volatility = self.data['atr'].iloc[self.current_step - 1]
        reward += profit / (volatility + 1e-8)
        if profit > 0:
            reward += 0.1
        elif profit < 0:
            reward -= 0.1
        reward += 0.01
        logging.debug(f"Прибыль: {profit}, Волатильность: {volatility}, Награда: {reward}")
        self.previous_balance = self.balance
        obs = self.normalized_data.iloc[self.current_step]
        self.obs_window.append(obs.values.astype(np.float32))
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True
            logging.info("Достигнут конец данных")
            # Пишем результат в df
            if self.is_backtest:
                create_result_df(self.result_df, STATS_DIR)

        self.balance_history.append(self.balance)
        if self.detect_error():
            self.handle_error()
            reward -= 10
            self.done = False
        return self._get_observation(), reward, self.done, False, info

    def _open_position(self, position_type, price, timestamp, atr):
        self.position = position_type
        self.entry_price = price
        self.entry_step = self.current_step
        self.position_size = self.balance * self.risk_percentage
        self.units = self.position_size / price
        # Удаляем переменные тейк-профита и стоп-лосса
        # self.take_profit_multiplier = 2
        # self.stop_loss_multiplier = 2
        self.positions.append({
            'entry_time': timestamp,
            'entry_price': price,
            'entry_step': self.current_step,
            'atr': atr
        })
        logging.info(
            f"Позиция открыта: {position_type} по цене {price}, Время: {timestamp}"
        )
        if self.is_backtest:
            self.result_df.loc[
                len(self.result_df), ["type", "open_ts", "open_price"]
            ] = [position_type, timestamp, price]

    def _close_position(self, price, timestamp):
        if self.entry_price == 0:
            logging.warning("Попытка закрыть позицию без входной цены")
            return 0
        fee_rate = 0.001
        slippage = 0.001
        duration = self.current_step - self.entry_step
        atr = self.data['atr'].iloc[self.entry_step]
        if self.position == 'long':
            effective_price = price * (1 - slippage)
            profit = (effective_price - self.entry_price) * self.units
        else:
            effective_price = price * (1 + slippage)
            profit = (self.entry_price - effective_price) * self.units
        fee = self.position_size * fee_rate * 2
        profit -= fee
        self.balance += profit
        self.total_profit += profit
        reward = profit / self.position_size
        reward = reward * 100
        # Удаляем влияние тейк-профита и стоп-лосса на награду
        # if self.position == 'long' and profit < 0:
        #     reward -= 0.1
        # if duration <= self.short_term_threshold and profit > 0:
        #     reward += 0.05
        # if profit > self.take_profit_multiplier * atr:
        #     reward += 0.1
        # elif profit < -self.stop_loss_multiplier * atr:
        #     reward -= 0.1
        # if duration <= self.short_term_threshold and profit > 0:
        #     reward += 0.05
        # if duration > self.long_term_threshold and profit < 0:
        #     reward -= 0.05
        self.positions[-1].update({
            'exit_time': timestamp,
            'exit_price': price,
            'duration': duration,
            'profit': profit,
            'atr': atr
        })
        logging.info(
            f"Позиция закрыта: {self.position} по цене {price}, Прибыль: {profit}, Награда: {reward}, Время: {timestamp}, Продолжительность: {duration}"
        )
        # Пишем результат в df
        if self.is_backtest:
            self.result_df.loc[
                len(self.result_df) - 1, ["close_ts", "close_price", "duration", "profit", "balance"]
                ] = [ timestamp, price, round((duration),0), profit, self.balance]

        self.position = None
        self.entry_price = 0
        self.position_size = 0
        self.units = 0
        return reward


def calculate_rvi(df, window=10):
    close_open = df['close'] - df['open']
    high_low = df['high'] - df['low']
    rvi = close_open / high_low
    rvi = rvi.rolling(window=window).mean()
    logging.debug("RVI рассчитан")
    return rvi


# Функция для добавления технических индикаторов
def add_technical_indicators(df):
    
    if MODEL_SUFFIX[-2:] == "V0":
        df = add_technical_indicators_v0(df) #empty
    elif MODEL_SUFFIX[-2:] == "V1":
        df = add_technical_indicators_original(df) 
    elif MODEL_SUFFIX[-2:] == "V2":
        df = add_technical_indicators_v2(df) #v1
    elif MODEL_SUFFIX[-2:] == "V3":
        df = add_technical_indicators_v3(df) #Bill Williams
    else:
        logging.error("Нужен суффикс модели")  
        return None 

    return df


# Original fn
"""async def get_full_data(exchange, symbol, timeframe=TIMEFRAME, since=None, limit=2016):  # Изменено с '1m' на '5m'
    all_ohlcv = []
    logging.info(f"Начало получения данных для символа {symbol}")
    while True:
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            if not ohlcv:
                logging.debug("Нет новых данных для загрузки")
                break
            all_ohlcv.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 5 * 60 * 1000  # Изменено с 60 секунд на 5 минут
            if last_timestamp >= exchange.milliseconds():
                logging.debug("Достигнута текущая временная метка")
                break
        except Exception as e:
            logging.error(f"Ошибка при получении данных: {e}\n", exc_info=True)
            break
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    logging.info(f"Получено {len(df)} записей данных для символа {symbol}")
    return df"""


def get_or_train_model_sync(symbol, train_df, models_dir, best_params=None):
    logging.info(f"Получение или обучение модели для символа {symbol}")

    model_path = f'{models_dir}/{symbol.replace("/", "_").split(":")[0]}'
    norm_path = f'{model_path}_norm.json'

    env = TradingEnvironment(train_df)
    norm_params = {'means': env.means.to_dict(), 'stds': env.stds.to_dict()}
    if os.path.exists(f"{model_path}.zip"):
        logging.info("Нахождение существующей модели, загрузка модели")
        if os.path.exists(norm_path):
            with open(norm_path, 'r') as f:
                norm_params = json.load(f)
        env = TradingEnvironment(train_df, norm_params=norm_params)
        env = DummyVecEnv([lambda: env])
        model = PPO.load(model_path, env=env, device="cpu")
        logging.info("Модель загружена успешно")
    else:
        logging.info("Модель не найдена, начало обучения")
        env = TradingEnvironment(train_df)
        means = env.means.to_dict()
        stds = env.stds.to_dict()
        env = DummyVecEnv([lambda: env])
        if best_params:
            net_arch = []
            n_layers = best_params.get('n_layers', 1)
            for i in range(n_layers):
                layer_size = best_params.get(f'n_units_l{i}', 64)
                net_arch.append(layer_size)
            activation = best_params.get('activation', 'tanh')
            activation_mapping = {
                'relu': torch.nn.ReLU,
                'tanh': torch.nn.Tanh,
                'elu': torch.nn.ELU
            }
            activation_fn = activation_mapping.get(activation, torch.nn.Tanh)
            policy_kwargs = dict(
                net_arch=dict(pi=net_arch, vf=net_arch),
                activation_fn=activation_fn
            )
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=best_params["learning_rate"],
                n_steps=best_params["n_steps"],
                gamma=best_params["gamma"],
                ent_coef=best_params["ent_coef"],
                vf_coef=best_params["vf_coef"],
                max_grad_norm=best_params["max_grad_norm"],
                policy_kwargs=policy_kwargs,
                tensorboard_log=f"./tensorboard_{MODEL_SUFFIX}/",
                verbose=1,
                device="cpu",
            )
        else:
            model = PPO(
                "MlpPolicy",
                env,
                tensorboard_log=f"./tensorboard_{MODEL_SUFFIX}/",
                verbose=1,
                device="cpu",
            )
        TIMESTEPS = 10000
        for i in range(1, 50):
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
            model.save(f"{model_path}_{TIMESTEPS*i}")     
        model.save(model_path)    
        with open(norm_path, 'w') as f:
            json.dump(norm_params, f)
        logging.info("Модель обучена и сохранена")
    return model, norm_params

def backtest_model_sync(model, test_df, symbol, norm_params):
    logging.info(f"Начало бэктеста модели для символа {symbol}")
    test_env = TradingEnvironment(test_df, norm_params=norm_params, is_backtest=True)
    obs, _ = test_env.reset()
    while not test_env.done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = test_env.step(action)
    logging.info("Бэктест завершен")

def objective_sync(trial, train_df, test_df):
    model_path = f'{MODELS_DIR}_{MODEL_SUFFIX}/{SYMBOL.replace("/", "_").split(":")[0]}'
    try:
        logging.debug(f"Начало оптимизации trial {trial.number}")
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
        n_steps = trial.suggest_categorical('n_steps', [128, 256])
        gamma = trial.suggest_float('gamma', 0.95, 0.99)
        
        ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-2, log=True)
        vf_coef = trial.suggest_float('vf_coef', 0.1, 1.0)
        max_grad_norm = trial.suggest_float('max_grad_norm', 0.3, 5.0)
        n_layers = trial.suggest_int('n_layers', 1, 3)
        net_arch = []
        for i in range(n_layers):
            layer_size = trial.suggest_int(f'n_units_l{i}', 64, 512)
            net_arch.append(layer_size)
        activation = trial.suggest_categorical('activation', ['tanh', 'relu', 'elu'])
        activation_mapping = {
            'relu': torch.nn.ReLU,
            'tanh': torch.nn.Tanh,
            'elu': torch.nn.ELU
        }
        activation_fn = activation_mapping.get(activation, torch.nn.Tanh)
        policy_kwargs = dict(
            net_arch=dict(pi=net_arch, vf=net_arch),
            activation_fn=activation_fn
        )
        env = TradingEnvironment(train_df)
        means = env.means.to_dict()
        stds = env.stds.to_dict()
        env = DummyVecEnv([lambda: env])
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=f"./tensorboard_{MODEL_SUFFIX}/",
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=100000)
        test_env = TradingEnvironment(test_df, norm_params={'means': means, 'stds': stds})
        obs, _ = test_env.reset()
        total_reward = 0
        while not test_env.done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = test_env.step(action)
            total_reward += reward
        env.close()
        test_env.close()
        logging.debug(f"Trial {trial.number} завершен с наградой {total_reward}")
        model.save(f"{model_path}_rew_{round(float(total_reward),2)}")  
        logging.debug(f"Модель Trial {trial.number} сохранена: {model_path}_rew_{total_reward}.zip")
        return total_reward
    except Exception as e:
        logging.error(f"Ошибка в trial {trial.number}: {e}\n", exc_info=True)
        return float('-inf')

async def run_optuna(study, train_df, test_df, n_trials):
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor()
    # executor = ProcessPoolExecutor(max_workers=os.cpu_count())
    for _ in range(n_trials):
        trial = study.ask()
        score = await loop.run_in_executor(executor, objective_sync, trial, train_df, test_df)
        study.tell(trial, score)
    executor.shutdown(wait=True)

class LiveTradingState:
    def __init__(self, window_size=48):
        self.window_size = window_size
        self.data = deque(maxlen=window_size)
        self.balance_history = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.balance = None

    def update(self, new_row, current_balance, timestamp):
        self.data.append(new_row)
        self.balance_history.append(current_balance)
        self.timestamps.append(timestamp)
        self.balance = current_balance
        logging.debug("Состояние live_trading обновлено")

    def get_dataframe(self):
        if len(self.data) == self.window_size:
            df = pd.DataFrame(list(self.data))
            df['balance'] = list(self.balance_history)
            df['timestamp'] = pd.to_datetime(list(self.timestamps))
            logging.debug("DataFrame для live_trading готов")
            return df
        else:
            logging.debug("Недостаточно данных для формирования DataFrame")
            return None

async def live_trading(async_exchange, model, symbol, norm_params, state):
    trading_interval = 300  # Изменено с 60 на 300 секунд (5 минут)
    logging.info("Запуск live_trading")
    while True:
        try:
            real_balance = await get_real_balance_async(async_exchange)
            logging.debug(f"Текущий баланс: {real_balance}")
            if real_balance is None:
                logging.warning("Не удалось получить баланс, ожидание перед следующей попыткой")
                await asyncio.sleep(trading_interval)
                continue
            ohlcv = await async_exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=1)  # Изменено с '1m' на '5m'
            if not ohlcv:
                logging.warning("Не удалось получить новые данные OHLCV")
                await asyncio.sleep(trading_interval)
                continue
            df_new = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms')
            timestamp = df_new['timestamp'].iloc[0]
            state.update(df_new.iloc[0], real_balance, timestamp)
            df = state.get_dataframe()
            if df is None:
                await asyncio.sleep(trading_interval)
                continue
            df = add_technical_indicators(df)
            if not all(feat in df.columns for feat in norm_params['means'].keys()):
                logging.warning("Недостающие признаки после добавления индикаторов")
                await asyncio.sleep(trading_interval)
                continue
            try:
                normalized_df = (df.drop(columns=['timestamp', 'balance']) - pd.Series(norm_params['means'])) / pd.Series(norm_params['stds'])
                obs = normalized_df.values.flatten().astype(np.float32)
                logging.debug(f"Наблюдение сформировано: {obs.shape}")
            except Exception as e:
                logging.error(f"Ошибка при нормализации данных: {e}\n", exc_info=True)
                await asyncio.sleep(trading_interval)
                continue
            if obs.shape[0] != model.observation_space.shape[0]:
                logging.error(f"Неожиданная форма наблюдения {obs.shape}, ожидается {model.observation_space.shape}")
                await asyncio.sleep(trading_interval)
                continue
            action, _states = model.predict(obs, deterministic=True)
            logging.debug(f"Предсказанное действие: {action}")
            positions = await async_exchange.fetch_positions(symbol)
            has_position = False
            current_contracts = 0
            current_side = None
            entry_price = 0
            if positions and isinstance(positions, list):
                for position in positions:
                    if position and 'contracts' in position and float(position.get('contracts', 0)) > 0:
                        has_position = True
                        current_contracts = float(position['contracts'])
                        current_side = position.get('side', '').lower()
                        entry_price = float(position.get('entryPrice', 0))
                        break
            current_price = float(df['close'].iloc[-1])
            amount = real_balance * 1.0 / current_price
            atr = df['atr'].iloc[-1]
            if action == 1:
                if has_position and current_side in ['sell', 'short']:
                    close_side = 'buy'
                    logging.info("Переключение с шорт на лонг: закрытие текущей позиции")
                    order = await async_exchange.create_order(symbol=symbol, type='market', side=close_side, amount=current_contracts)
                if not has_position or current_side != 'long':
                    logging.info("Открытие длинной позиции")
                    order = await async_exchange.create_order(symbol=symbol, type='market', side='buy', amount=amount)
            elif action == 2:
                if has_position and current_side in ['buy', 'long']:
                    close_side = 'sell'
                    logging.info("Переключение с лонг на шорт: закрытие текущей позиции")
                    order = await async_exchange.create_order(symbol=symbol, type='market', side=close_side, amount=current_contracts)
                if not has_position or current_side != 'short':
                    logging.info("Открытие короткой позиции")
                    order = await async_exchange.create_order(symbol=symbol, type='market', side='sell', amount=amount)
            # Действие 0: удерживать позицию, никаких действий не требуется
        except Exception as e:
            logging.error(f"Ошибка в live_trading: {e}\n", exc_info=True)
        await asyncio.sleep(trading_interval)

async def main():
    if LOGS_TO_FILE=="NEW":  
        clear_log_file(log_filename) # Удаляем логи предыдущего запука
    loop = asyncio.get_running_loop()
    if not sys.platform.startswith('win'):
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, shutdown_handler)
    async_exchange = ccxt_async.bybit(exchange_config)
    if ACC_MODE == "DEMO":  
        async_exchange.enable_demo_trading(True)  # Включаем режим демо-счета

    # Получение баланса для проверки подключения
    balance = await async_exchange.fetch_balance()
    usdt_balance = balance["total"].get("USDT", 0)
    logging.debug(f"Текущий баланс: {usdt_balance} USDT")   
    # await is_continue(async_exchange)

    executor = ThreadPoolExecutor()
    try:
        symbol = SYMBOL  
        models_dir = f"{MODELS_DIR}_{MODEL_SUFFIX}"    
        os.makedirs(models_dir, exist_ok=True)

        model_path = f'{models_dir}/{symbol.replace("/", "_").split(":")[0]}'
        norm_path = f'{model_path}_norm.json'

        if not await verify_symbol(async_exchange, symbol):
            logging.error(f"Символ {symbol} недоступен")
            return
        
        df = await get_full_data(async_exchange, symbol, timeframe=TIMEFRAME, limit=GET_DATA_LIMIT)   # Изменено на '5m' 
        if df is not None and not df.empty:
            df = add_technical_indicators(df)
            train_size = int(len(df) * 0.8)
            train_df = df.iloc[:train_size].reset_index(drop=True)
            test_df = df.iloc[train_size:].reset_index(drop=True)

            if RUN_MODE == "LEARN": # Обучение
                study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
                await run_optuna(study, train_df, test_df, n_trials=15)
                best_params = study.best_params
                logging.info(f"Лучшие параметры оптимизации: {best_params}")
                with open(f'{model_path}_best_param.json', "w") as f:
                    json.dump(best_params, f)
                logging.info(f"Лучшие параметры оптимизации сохранены в : {best_params}")
                model, norm_params = await loop.run_in_executor(executor, get_or_train_model_sync, symbol, train_df, models_dir, best_params)

            if RUN_MODE == "TRADE_ONLY" or RUN_MODE == "BK":  # Сразу загружаем модель
                with open(norm_path, 'r') as f:
                    norm_params = json.load(f)
                logging.info("Нормализованные параметры загружены успешно")
                # logging.info(f"{norm_params=}")
                model = PPO.load(model_path, device="cpu")
                logging.info(f"Модель {model_path}.zip загружена успешно" )
                # await is_continue(exchange=async_exchange)

            if BK_DIR:  
                file_path = create_file_path(symbol, timeframe=TIMEFRAME, data_dir=BK_DIR)  
                logging.info(f"Для Бэктеста используется файл: {file_path}")
                df = pd.read_csv(file_path)
                mask = create_date_mask(df, start_date=BK_START, end_date=BK_END)  
   
                test_df = df[mask]
                logging.info(f"Бэктест с {test_df["timestamp"].iloc[1]} по {test_df["timestamp"].iloc[-1]}")
                
                # await is_continue(exchange=async_exchange)
                
                if test_df is not None and not test_df.empty:
                    test_df = add_technical_indicators(test_df)
                    test_df = test_df.reset_index(drop=True)
                else:
                    logging.error("Не удалось загрузить данные или данные пусты")
                    await is_continue(exchange=async_exchange, exit=True)
                # await is_continue(exchange=async_exchange)

            await loop.run_in_executor(executor, backtest_model_sync, model, test_df, symbol, norm_params)
            await is_continue(exchange=async_exchange, exit=True)

            if RUN_MODE == "TRADE_ONLY":

                state = LiveTradingState(window_size=48)
                last_20_data = test_df.tail(state.window_size)
                initial_balance = await get_real_balance_async(async_exchange)
                if initial_balance is None:
                    initial_balance = 10
                for _, row in last_20_data.iterrows():
                    state.update(row, initial_balance, row['timestamp'])
                await live_trading(async_exchange, model, symbol, norm_params, state)
        else:
            logging.error("Не удалось загрузить данные или данные пусты")
    except asyncio.CancelledError:
        logging.info("Задачи были отменены")
    except Exception as e:
        logging.error(f"Ошибка в main: {e}\n", exc_info=True)
    finally:
        await async_exchange.close()
        executor.shutdown(wait=True)
        logging.info("Обмен закрыт и исполнитель завершен")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Программа прервана пользователем")
