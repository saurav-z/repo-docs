# ChatGPT Micro-Cap Experiment: Can AI Beat the Market?

**Dive into a real-world experiment where ChatGPT manages a micro-cap stock portfolio, testing the power of AI in financial markets.** ([Original Repo](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment))

## Project Overview

This project documents a six-month live trading experiment using ChatGPT to manage a real-money micro-cap portfolio. The goal? To determine if a powerful large language model can generate alpha or make smart trading decisions using real-time data.

## Key Features

*   **AI-Powered Trading:** Leverages ChatGPT-4 to make trading decisions.
*   **Automated Trading Engine:** Includes a Python-based trading script with portfolio management and stop-loss automation.
*   **Daily Performance Tracking:**  Monitors and publishes daily portfolio performance, including PnL, total equity, and trade history.
*   **Transparent Data:** Offers access to detailed trade logs and performance data via CSV files.
*   **Research & Analysis:** Provides weekly deep research summaries, performance reports, and experiment methodology.
*   **Visualization:** Uses Matplotlib to create graphs comparing ChatGPT's performance to market benchmarks.

## Repository Structure

*   **`trading_script.py`**: The core trading engine responsible for portfolio management and stop-loss automation.
*   **`Scripts and CSV Files/`**: Contains the experiment's real-time portfolio data, updated daily.
*   **`Start Your Own/`**: Provides template files and a guide to help you replicate the experiment.
*   **`Weekly Deep Research (MD|PDF)/`**: Includes research summaries and performance reports.
*   **`Experiment Details/`**: Offers in-depth documentation, methodology, prompts, and Q&A.

## The Concept

Inspired by the growing hype around AI in finance, this project sets out to answer a fundamental question: Can a large language model, like ChatGPT, make profitable trading decisions? The experiment uses a $100 starting budget, with strict stop-loss rules and weekly deep research sessions, to test this hypothesis in a real-world setting.

## Performance & Results

![Latest Performance Results](Results.png)

**Current Status:** Portfolio is underperforming the S&P 500 benchmark.

*Performance data is updated after each trading day. See the CSV files in `Scripts and CSV Files/` for detailed daily tracking.*

## Core Technologies

*   **Python**: Core scripting and automation
*   **pandas + yFinance**: Market data fetching and analysis
*   **Matplotlib**: Performance visualization and charting
*   **ChatGPT-4**: AI-powered trading decision engine

## Additional Features

*   **Robust Data Sources**: Uses Yahoo Finance as the primary data source, with Stooq as a fallback for reliability.
*   **Automated Stop-Loss**: Automatic position management with configurable stop-losses.
*   **Interactive Trading**: Includes Market-on-Open (MOO) and limit order support.
*   **Backtesting Support**: ASOF_DATE override for historical analysis.
*   **Performance Analytics**: CAPM analysis, Sharpe/Sortino ratios, and drawdown metrics.
*   **Trade Logging**: Complete transparency with detailed execution logs.

## System Requirements

*   Python 3.11+
*   Internet connection for market data
*   ~10MB storage for CSV data files

## Follow Along

The experiment runs from June 2025 to December 2025. The portfolio CSV file will be updated daily.

Weekly updates are posted on the project's blog: [A.I Controls Stock Account](https://nathanbsmith729.substack.com)

## Contribute

Contributions are highly encouraged!

*   **Issues:** Report bugs or suggest improvements.
*   **Pull Requests:** Submit your contributions.
*   **Collaboration:** High-value contributors may be invited as maintainers/admins.

For more information, check out the [Contributing Guide](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/blob/main/Other/CONTRIBUTING.md).

## Contact

Have feature requests or any advice?

Please reach out here: **nathanbsmith.business@gmail.com**