# Synapse Trading Bot: Automate Your Crypto Trading with AI 🤖

**Synapse is an open-source intelligent trading bot designed to help you navigate the crypto market, offering automated trading, backtesting, and real-time insights.  View the original repository [here](https://github.com/anthugeist/synapse-trading-bot).**

[![Version](https://img.shields.io/badge/version-0.12.6-blue)](https://github.com/anthugeist/synapse-trading-bot)
[![Coverage](https://img.shields.io/badge/coverage-75%25-yellowgreen)](https://github.com/anthugeist/synapse-trading-bot)

---

## Key Features

*   **Automated Trading Signals:** Generates buy/sell signals and executes trades based on real-time market analysis.
*   **Backtesting:** Evaluate trading strategies using historical data to optimize performance without risking capital.
*   **AI-Powered Strategy Optimization:**  Receive AI-driven recommendations to enhance your custom trading strategies.
*   **Self-Training:** Continuously learns and adapts, improving its trading strategies over time based on user results and data updates.
*   **Telegram Integration:** Monitor trades and control the bot seamlessly through a user-friendly Telegram interface.
*   **Manual Trading Mode:** Snipe tokens with precision, using custom stop-loss and take-profit levels.
*   **Price Prediction:** Forecasts token prices by analyzing market trends, sentiment, and historical patterns.
*   **Financial Management:** Tracks cumulative profit/loss in real-time and provides risk alerts.
*   **Supported Exchanges:** Supports a wide array of exchanges (logos included in original README - not repeated for brevity).

---

## Installation

### Requirements:

*   Python 3.10+
*   Git
*   Pip
*   2v CPU, 2GB DDR4, 2GB disk space

### Steps:

**Option 1 (using `run.bat` - if available):**

```bash
git clone https://github.com/anthugeist/synapse-trading-bot
cd synapse-trading-bot
run.bat
```

**Option 2 (Manual):**

```bash
git clone https://github.com/anthugeist/synapse-trading-bot
cd synapse-trading-bot
pip install -r requirements.txt
python bot.py
```

### Connecting to Telegram:

1.  Enter your bot token and chat ID in `config.json`.
2.  Run the bot with the Telegram flag:

```bash
cd synapse-trading-bot
python bot.py --telegram
```

### Telegram Commands:

*   `/start`: Starts the bot
*   `/stop`: Stops the bot
*   `/stats`: Displays account statistics (balance, trade history, profit)
*   `/mode`: Switches trading mode
*   `/status`: Shows current operating mode
*   `/autotrade_on`: Enables automated trading
*   `/autotrade_off`: Disables automated trading
*   `/snipe TOKEN`: Buys a token instantly or at a specific price
*   `/sell TOKEN`: Sells a token instantly
*   `/strategy_suggest`: Gets AI-driven strategy recommendations
*   `/backtest *strategy name*`: Runs a backtesting simulation
*   `/help`: Displays the full list of commands

*Note: This is not an exhaustive list.  Use `/help` in the bot for a complete list.*

---

## Disclaimer

Use Synapse at your own risk. The developers are not responsible for your trading outcomes. Only trade with funds you can afford to lose.

---

## Support & Community

*   **Telegram Channel:** [![Telegram Channel](https://img.shields.io/badge/Telegram-Channel-Link?style=for-the-badge&logo=Telegram&logoColor=white&logoSize=auto&color=blue)](https://t.me/+pB6j65Kv7cdjZmU0)
*   **Support/Bug Reports:** Join the [Telegram Community](https://t.me/+9j5RcKMfT5s4M2Q0)
*   **Suggestions:** Contact [@Hhubbinmo3](https://t.me/@Hhubbinmo3) on Telegram.
*   **Bitcoin Address:** `bc1q3zykl6k0jyk864kdeqdfq6hudfr3ry8wksrr6u` (For tips)

---

**Star the Repo! ⭐**