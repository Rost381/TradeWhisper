import os
import logging
import ccxt.async_support as ccxt_async
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load API credentials from environment variables
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
PASSPHRASE = os.getenv("PASSPHRASE")

# Configure the OKX exchange connection
exchange_config = {
    "apiKey": API_KEY,
    "secret": SECRET_KEY,
    "password": PASSPHRASE,  # OKX uses 'password' for the passphrase
    "enableRateLimit": True,
    "mode": "demo",  # Set to 'demo' for demo trading
    "options": {
        "defaultType": "swap",  # Set to 'swap' for futures
        "adjustForTimeDifference": True,
        "recvWindow": 10000,
        "mode": "demo",  # Set to 'demo' for demo trading
    },
    "timeout": 30000,
}


# Initialize the OKX exchange
async def initialize_exchange():
    exchange = ccxt_async.okx(exchange_config)
    await exchange.load_markets()
    return exchange


# Example function to fetch OHLCV data for a specific symbol
async def fetch_ohlcv(exchange, symbol, timeframe="5m", limit=2000):
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return ohlcv
    except Exception as e:
        logging.error(f"Error fetching OHLCV data: {e}")
        return []


# Example function to get balance
async def get_balance(exchange):
    try:
        balance = await exchange.fetch_balance()
        return balance
    except Exception as e:
        logging.error(f"Error fetching balance: {e}")
        return None


# Example function to create an order
async def create_order(exchange, symbol, order_type, side, amount, price=None):
    try:
        if order_type == "market":
            order = await exchange.create_market_order(symbol, side, amount)
        elif order_type == "limit":
            order = await exchange.create_limit_order(symbol, side, amount, price)
        return order
    except Exception as e:
        logging.error(f"Error creating order: {e}")
        return None


# Example usage
async def main():
    
    exchange = await initialize_exchange()
    balance = await get_balance(exchange)
    logging.info(f"Balance: {balance}")

    # Close the exchange connection
    await exchange.close()


# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
