import ccxt
import os
from dotenv import load_dotenv

def get_bybit_demo_balance():
    # Загрузка API-ключей из файла .env
    load_dotenv()
    api_key = os.getenv("BYBIT_API_KEY")
    secret_key = os.getenv("BYBIT_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError("API ключи не найдены. Проверьте файл .env.")

    # Инициализация Bybit с демо-сервером
    bybit = ccxt.bybit(
        #     {
        #     "apiKey": api_key,
        #     "secret": secret_key,
        # }
        {
            "apiKey": api_key,
            "secret": secret_key,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                "adjustForTimeDifference": True,
                "recvWindow": 10000,
            },
            "timeout": 30000,
        }
    )
    bybit.enable_demo_trading(True)  # Включаем режим демо-счета

    try:
        # Получение баланса
        balance = bybit.fetch_balance()
        usdt_balance = balance["total"].get("USDT", 0)
        btc_balance = balance["total"].get("BTC", 0)
        return usdt_balance, btc_balance
    # balance
    except ccxt.BaseError as e:
        print(f"Ошибка при получении баланса: {e}")
        return None

# Пример использования
if __name__ == "__main__":
    demo_balance = get_bybit_demo_balance()
    if demo_balance:
        print("Баланс на демо-счете Bybit:")
        print(demo_balance)
