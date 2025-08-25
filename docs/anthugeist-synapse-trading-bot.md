# Synapse Trading Bot: Your AI-Powered Crypto Trading Companion

**Synapse is an open-source trading bot that uses real-time market data, social sentiment, and AI to generate trading signals and automate your crypto trading strategies.**  For more details and to explore the project, visit the original repository: [Synapse Trading Bot on GitHub](https://github.com/anthugeist/synapse-trading-bot).

## Key Features

*   **AI-Driven Trading Signals:** Automatically generates buy/sell signals based on live market conditions, optimizing stop-loss and take-profit levels.
*   **Backtesting for Strategy Optimization:** Test your strategies on historical data to refine parameters and evaluate performance risk-free.
*   **Self-Learning AI:** Synapse continuously learns from your trading results, adapting strategies and improving over time.
*   **Custom Strategy Analysis:** Machine learning analyzes your custom trading strategies, providing insights and suggestions for improvement.
*   **Seamless Integration & Intuitive Control:** Manage trades easily through a Telegram bot or web UI.
*   **Manual Mode for Precision:** Snipe tokens, set custom stop-loss/take-profit levels, and execute your own strategies.
*   **Real-Time Notifications:** Stay informed with Telegram updates throughout the trading process.
*   **Price Prediction:** Forecast token prices using market trends, the Fear and Greed Index, and historical patterns.
*   **Financial Management:** Monitor your trading performance, tracking cumulative profit and loss in real time, and receive risk alerts.

## Supported Exchanges

<div align="center">
<!-- Insert exchange logos here (as per the original README) -->
<img width="120" height="100" alt="image" src="https://github.com/user-attachments/assets/90d8ca5a-71d8-404d-80a4-578e1efe2db9" />  <img width="120" height="100" alt="image" src="https://github.com/user-attachments/assets/900cb4eb-8d14-4b51-97b1-3c515ea60141" />  <img width="120" height="100" alt="image" src="https://github.com/user-attachments/assets/cdc3f7ef-6ad4-423c-b0bc-8669761774db" />  <img width="120" height="100" alt="image" src="https://github.com/user-attachments/assets/faec88fc-4946-48e7-b98f-157e2234e7f8" />  <img width="120" height="100" alt="image" src="https://github.com/user-attachments/assets/0f836b8b-5f4b-4ddb-a1d2-7318d883d51f" />  <img width="120" height="100" alt="image" src="https://github.com/user-attachments/assets/29df4163-dd45-42ff-96dc-aec0e5d1788a" />  <img width="120" height="100" alt="image" src="https://github.com/user-attachments/assets/8c1e6327-f80d-42a7-a918-5fd5c4daf441" />  <img width="120" height="100" alt="image" src="https://github.com/user-attachments/assets/f4008a15-a371-4bc5-9ca0-95839eb2afbf" />  <img width="120" height="100" alt="image" src="https://github.com/user-attachments/assets/eaafbfe9-1359-4068-a321-b4d982739edf" />  <img width="120" height="100" alt="image" src="https://github.com/user-attachments/assets/492b1317-cbdf-4a17-b67e-eebbe47a4315" />  <img width="120" height="100" alt="image" src="https://github.com/user-attachments/assets/31fbf32d-dfbb-4bd8-b10d-782c6fc3a74f" />  <img width="120" height="100" alt="image" src="https://github.com/user-attachments/assets/9b61064c-e4a7-4458-9331-c04c746cf5b8" />  <img width="120" height="100" alt="image" src="https://github.com/user-attachments/assets/c9b41a39-344d-4163-84b5-d281c1c5c9a2" />  <img width="120" height="100" alt="image" src="https://github.com/user-attachments/assets/b21dddf6-5787-4da0-a2d9-440e91f71dd2" />  <img width="120" height="100" alt="image" src="https://github.com/user-attachments/assets/73741faf-0aa5-4967-bbea-ec101da6b9d2" />  <img width="120" height="100" alt="image" src="https://github.com/user-attachments/assets/2af7c9c2-5bf1-4770-accd-785e87c1fb51" />
</div>

## Installation

**Prerequisites:**

*   Python 3.10+
*   Git
*   Pip
*   2v CPU, 2GB DDR4, 2GB disk space

**Steps:**

```bash
git clone https://github.com/anthugeist/synapse-trading-bot
cd synapse-trading-bot
run.bat
```

**OR**

```bash
git clone https://github.com/anthugeist/synapse-trading-bot
cd synapse-trading-bot
pip install -r requirements.txt
python bot.py
```

To connect with the Telegram bot:

1.  Populate your bot token and chat ID in `config.json`.
2.

    ```bash
    cd synapse-trading-bot
    python bot.py --telegram
    ```
3.  Use these RPC commands to control the bot via Telegram:

| Command          | Description                                                                        |
| ---------------- | ---------------------------------------------------------------------------------- |
| `/start`         | Start the bot                                                                     |
| `/stop`          | Stop the bot                                                                      |
| `/stats`         | Show account statistics (balance, trade history, overall profit)                 |
| `/mode`          | Switch trading mode                                                              |
| `/status`        | Show current operating mode                                                      |
| `/autotrade_on`  | Enable automatic trading based on generated signals                            |
| `/autotrade_off` | Disable auto trading                                                              |
| `/snipe TOKEN`   | Buy a selected token instantly at market price or snipe at a desired price       |
| `/sell TOKEN`    | Sell a selected token instantly at market price                                 |
| `/strategy_suggest` | AI-analyzed strategy recommendations based on your performance                 |
| `/backtest *strategy name*` | Run a backtesting simulation for a saved strategy                        |
| `/help`          | Show the full list of commands                                                   |

*   *Note: This is not the full list of RPC commands. Type `/help` in the bot for a complete list.*

## Disclaimer

Use this bot at your own risk. The developer is not responsible for your trading results. Only risk capital you can afford to lose. Trade wisely.

## Support

***Support/Bug Reports***

[![Telegram Channel](https://img.shields.io/badge/Telegram-Channel-Link?style=for-the-badge&logo=Telegram&logoColor=white&logoSize=auto&color=blue)](https://t.me/+pB6j65Kv7cdjZmU0)

Encountered any difficulties or bugs? Join our Telegram [community](https://t.me/+9j5RcKMfT5s4M2Q0)

Have ideas for improvement? Contact the developer on [Telegram](https://t.me/@Hhubbinmo3)

***Buy Me a Coffee***

BTC: `bc1q3zykl6k0jyk864kdeqdfq6hudfr3ry8wksrr6u`

**Don't forget to star the repository if you like it! ⭐**