import asyncio
import logging
import ccxt.async_support as ccxt_async
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "TAHqn***xGhVmof2E")
API_SECRET = os.getenv("API_SECRET", "0QVQhcRx6SP1uZ****BJDIiQKqN1")


if not API_KEY or not API_SECRET:
    raise ValueError("API ключи не найдены. Проверьте файл .env.")

# Инициализация Bybit с демо-сервером
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


async def get_real_balance(exchange):
    try:
        balance = await exchange.fetch_balance()
        real_balance = balance["total"].get("USDT", 0)
        logging.debug(f"Текущий баланс: {real_balance} USDT")
        return real_balance
    except Exception as e:
        logging.error(f"Ошибка при получении баланса: {e}\n", exc_info=True)
        return None
    finally:
        await exchange.close()  # Закрываем соединение


# Пример использования
async def main():
    async_exchange = ccxt_async.bybit(exchange_config)
    async_exchange.enable_demo_trading(True)  # Включаем режим демо-счета
    balance = await get_real_balance(async_exchange)
    print(f"Текущий баланс: {balance} USDT")


if __name__ == "__main__":
    asyncio.run(main())
