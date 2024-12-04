# version №4

import os
import asyncio
import logging
import json
import hmac
import base64
import time
import okx.Account as Account
import okx.Trade as Trade
import okx.PublicData as PublicData
import okx.MarketData as MarketData
import okx.Funding as Funding
import aiohttp

# Конфигурация API
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
PASSPHRASE = os.getenv("PASSPHRASE")
FLAG = "1"  # Режим демо-трейдинга


print(API_KEY, SECRET_KEY, PASSPHRASE)
input("-------------------")

# Инициализация API-клиентов
accountAPI = Account.AccountAPI(API_KEY, SECRET_KEY, PASSPHRASE, False, FLAG)
tradeAPI = Trade.TradeAPI(API_KEY, SECRET_KEY, PASSPHRASE, False, FLAG)
marketAPI = MarketData.MarketAPI(API_KEY, SECRET_KEY, PASSPHRASE, False, FLAG)


# Функция для получения баланса
async def fetch_balance():
    result = accountAPI.get_account_balance()
    print("Баланс аккаунта:", result)


# Функция для открытия позиции
async def open_position(symbol, side, amount, price=None):
    order_type = "limit" if price else "market"
    order = tradeAPI.place_order(
        instId=symbol,
        tdMode="cross",
        side=side,
        ordType=order_type,
        sz=str(amount),
        px=str(price) if price else None,
    )
    print("Ордер открыт:", order)


# Функция для закрытия позиции
async def close_position(symbol, order_id):
    result = tradeAPI.cancel_order(instId=symbol, ordId=order_id)
    print("Ордер закрыт:", result)


# Функция для подключения к WebSocket
async def okx_ws_connect():
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect("wss://ws.okx.com:8443/ws/v5/private") as ws:
            timestamp = str(int(time.time()))
            sign = hmac.new(
                SECRET_KEY.encode("utf-8"),
                f"{timestamp}GET/users/self/verify".encode("utf-8"),
                digestmod="sha256",
            ).digest()
            sign = base64.b64encode(sign).decode()

            login_msg = {
                "op": "login",
                "args": [
                    {
                        "apiKey": API_KEY,
                        "passphrase": PASSPHRASE,
                        "timestamp": timestamp,
                        "sign": sign,
                    }
                ],
            }
            await ws.send_json(login_msg)

            # Подписка на заказы
            await ws.send_json({"op": "subscribe", "args": [{"channel": "orders"}]})

            async for msg in ws:
                print("Сообщение WebSocket:", msg.json())


# Основная функция
async def main():
    # await fetch_balance()
    # await open_position("BTC-USDT-SWAP", "buy", 1)
    await okx_ws_connect()


if __name__ == "__main__":
    asyncio.run(main())
