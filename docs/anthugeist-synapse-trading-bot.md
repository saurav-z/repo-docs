# Synapse Trading Bot: Your AI-Powered Crypto Trading Companion

**Synapse is an open-source trading bot that uses real-time market data, social sentiment, and historical patterns to generate profitable trading signals, offering a smart and adaptable approach to crypto trading.** Explore the features and capabilities of Synapse and discover how it can enhance your trading strategy. [View the original repository](https://github.com/anthugeist/synapse-trading-bot) for more information.

## Key Features

*   **Automated Signal Generation:** Generates trading signals automatically based on live market conditions, and can optionally execute buy/sell orders, setting optimal stop-loss and take-profit levels.
*   **Backtesting:** Test your strategies on historical market data to evaluate performance, fine-tune parameters, and optimize results without risking real capital.
*   **AI-Powered Strategy Optimization:** Analyzes your custom trading strategies using machine learning and market data, providing targeted suggestions for improvement.
*   **Self-Training:** Continuously learns from your trading results, adapting its strategies and improving overall intelligence over time.
*   **Intuitive Interface:** Offers a simple interface with control via a Telegram bot or web UI for easy management.
*   **Manual Mode:** Snipe tokens at optimal prices with full control and precision, defining stop-loss, take-profit levels, and custom strategies.
*   **Real-time Notifications:** Stay informed with real-time updates via Telegram throughout the trading process.
*   **Price Prediction:** Forecasts individual token prices by analyzing current market trends, the Fear and Greed Index, and historical patterns.
*   **Financial Management:** Monitors overall trading performance, tracking cumulative profit and loss in real time, with risk alerts before trades.

## Supported Exchanges

\[*Images of supported exchanges -  as per original README*]

## Installation

**Requirements:**

*   Python 3.10+
*   Git
*   Pip
*   2v CPU, 2GB DDR4, 2GB disk space

**Installation Steps:**

**Option 1 (using `run.bat`)**

```shell
git clone https://github.com/anthugeist/synapse-trading-bot
cd synapse-trading-bot
run.bat
```

**Option 2 (manual)**

```shell
git clone https://github.com/anthugeist/synapse-trading-bot
cd synapse-trading-bot
pip install -r requirements.txt
python bot.py
```

**Connecting to Telegram:**

1.  Place your bot token and chat ID in `config.json`.
2.  Run the bot with the Telegram integration:

```shell
cd synapse-trading-bot
python bot.py --telegram
```

**Telegram RPC Commands:**

| Command                      | Description                                                                                                |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `/start`                     | Start the bot                                                                                              |
| `/stop`                      | Stop the bot                                                                                               |
| `/stats`                     | Show statistics of your account (balance, trade history, overall profit)                                    |
| `/mode`                      | Switch trading mode                                                                                          |
| `/status`                    | Show current operating mode                                                                                   |
| `/autotrade_on`              | Enable automatic trading based on generated signals                                                          |
| `/autotrade_off`             | Disable auto trading                                                                                         |
| `/snipe TOKEN`               | Buy a selected token instantly at market price or snipe at a desired price                                    |
| `/sell TOKEN`                | Sell a selected token instantly at market price                                                              |
| `/strategy_suggest`          | AI-analyzed strategy recommendations based on your performance                                                 |
| `/backtest *strategy name*` | Run backtesting simulation for a saved strategy                                                              |
| `/help`                      | Show full list of commands                                                                                  |

*Note: This is not an exhaustive list. Type `/help` in the bot for a complete command list.*

## Disclaimer

Use the bot at your own risk. The developer is not responsible for your trading outcomes. Only trade with funds you can afford to lose.

## Support

*   **Telegram Channel:** [![Telegram Channel](https://img.shields.io/badge/Telegram-Channel-Link?style=for-the-badge&logo=Telegram&logoColor=white&logoSize=auto&color=blue)](https://t.me/+pB6j65Kv7cdjZmU0)
*   **Community:** Join our [Telegram community](https://t.me/+9j5RcKMfT5s4M2Q0) for support and bug reports.
*   **Contact:** For improvement ideas, contact @Hhubbinmo3 on [Telegram](https://t.me/@Hhubbinmo3).

## Donate

If you appreciate the project, consider supporting it:

*   Bitcoin: `bc1q3zykl6k0jyk864kdeqdfq6hudfr3ry8wksrr6u`

**Don't forget to put stars!** ⭐