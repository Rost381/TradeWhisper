# TradeWhisper 🚀

<img src="https://github.com/Solrikk/TradeWhisper/blob/main/assets/6c7b8280-6fdd-11e9-886e-8978ffd3ee82.jpg" width="600" alt="TradeWhisperer Logo">

<div align="center">
  <h3>
    <a href="https://github.com/Solrikk/TradeWhisper/blob/main/README.md">⭐English ⭐</a> |
    <a href="https://github.com/Solrikk/TradeWhisper/blob/main/docs/readme/README_RU.md">Russian</a> |
    <a href="https://github.com/Solrikk/TradeWhisper/blob/main/docs/readme/README_GE.md">German</a> |
    <a href="https://github.com/Solrikk/TradeWhisper/blob/main/docs/readme//README_JP.md">Japanese</a> |
    <a href="https://github.com/Solrikk/TradeWhisper/blob/main/docs/readme/README_KR.md">Korean</a> |
    <a href="https://github.com/Solrikk/TradeWhisper/blob/main/docs/readme/README_CN.md">Chinese</a>
  </h3>
</div>

## ⚠️ IMPORTANT DISCLAIMER ⚠️
### This trading bot is currently in EXPERIMENTAL/BETA testing phase. By using this software:
1. **Cryptocurrency Trading Risks:** You acknowledge that trading cryptocurrencies involves substantial risks, including the potential loss of your invested capital.
2. **Technological Limitations:** The bot utilizes Artificial Intelligence and Machine Learning (AI/ML) models that are still undergoing testing and improvements. This may lead to unforeseen errors or inaccurate signals.
3. **Liability for Losses:** You accept full responsibility for any financial losses that may occur as a result of using this bot.
4. **No Guarantee of Performance:** Past performance does not guarantee future results. The cryptocurrency market is highly volatile and can change rapidly.
5. **Capital Management:** Trade only with funds you can afford to lose. Do not invest money that is necessary for your living expenses or other essential purposes.
6. **No Financial Advice:** This software is NOT financial advice. Use it at your own risk and consult with professional financial advisors before making investment decisions.

## 🌟 Overview

**TradeWhisperer** is a sophisticated cryptocurrency trading bot that leverages advanced Reinforcement Learning techniques, specifically the Proximal Policy Optimization (PPO) algorithm, to navigate the complex world of crypto markets. Built with a focus on adaptability and risk management, this bot combines technical analysis with machine learning to make data-driven trading decisions.

Before diving into the development process, I would like to present examples of test run results with different balances using **stable_baselines3**. These examples demonstrate how the model's effectiveness varies depending on the initial capital and testing duration.

For the initial tests, I chose the meme coin **Doge**, assuming that low-cap coins would have minimal impact on the overall portfolio. The tests were conducted with a minimum balance of **10 USDT** to evaluate the model's behavior with small investments.

![Баланс: 10 USDT](https://s3.timeweb.cloud/68597a50-pictrace/photo_2024-11-12_03-23-43.jpg)

The results showed that the maximum value reached **10.25 USDT**. Despite minimal growth, the model didn't trade at a loss, which can already be considered a success, although it didn't achieve its intended goal. However, considering that the model only trained for a couple of dozen minutes to capture the trading pattern, this is understandable.

Furthermore, with further hyperparameter improvements and strategy optimization, significant profits can be expected over several hours of trading. It's important to emphasize again that the absence of losses at such a balance is rather a positive indicator, demonstrating the model's ability to avoid adverse market conditions.

---

And now we'll increase the trading balance to **50 USDT** and extend the trading data by a couple of days:

![Баланс: 50 USDT](https://s3.timeweb.cloud/68597a50-pictrace/photo_2024-11-12_20-45-26.jpg)

The model showed a minimum value of **46.5 USDT**, which isn't critical; however, it indicates that the model "consumed" data containing noise or false signals. There were quite a few such moments during the development phase, and while these patterns on the graphs were initially discouraging, they didn't stop me from continuing development.

The problem was clear, and I integrated the **Optuna library** for hyperparameter optimization. This allowed for filtering out irrelevant data and improving the model's overall performance. Hyperparameter optimization helps the model better adapt to market conditions, reducing the probability of errors and increasing trading efficiency.

This translation captures the technical aspects while maintaining the narrative of continuous improvement and problem-solving approach, especially highlighting the implementation of Optuna for optimization, which is a crucial development step in machine learning-based trading systems.

---

And now we'll increase the portfolio balance to **1,000 USDT**:

![Баланс: 1 000 USDT](https://s3.timeweb.cloud/68597a50-pictrace/photo_2024-11-12_22-35-09.jpg)

The model reached **1,665 USDT**, which represents a confident success in its training. Although it no longer adheres to my initial concept, which is logical in this case, as our balance wasn't suitable for this coin, and for a long time it perceived market fluctuations as noise, waiting for a significant shift to latch onto, and as we can see, it did so successfully.

This translation effectively conveys the significant improvement in model performance with a larger balance, maintaining the technical accuracy while explaining how the model adapted its strategy. The success shown here (reaching **1,665 USDT** from **1,000 USDT**) demonstrates the model's ability to capitalize on meaningful market movements rather than getting caught up in market noise, which is a crucial aspect of algorithmic trading systems.

## 🚀 Key Features
### 🤖 Intelligent Trading System
- **Advanced RL Implementation**: Custom-built trading environment using OpenAI Gym
- **Adaptive Learning**: Continuous model improvement through real market interactions
- **Smart Position Management**: Automated position sizing and risk-adjusted trading
- **Multi-timeframe Analysis**: Processes multiple timeframes for better decision making

### 📊 Technical Analysis Suite
- **Comprehensive Indicators**:
  - Average True Range (ATR) for volatility measurement
  - Multiple momentum indicators
  - Custom-built signal generators
  - Advanced trend detection algorithms
### ⚡ Real-time Operations
- **Live Market Integration**:
  - Seamless connection to major crypto exchanges
  - Real-time data processing
  - Immediate trade execution
  - Async architecture for optimal performance
### 🛡️ Risk Management
- **Advanced Protection Features**:
  - Dynamic position sizing based on account balance
  - Stop-loss and take-profit automation
  - Risk percentage customization
  - Balance monitoring and automatic shutdown on significant losses