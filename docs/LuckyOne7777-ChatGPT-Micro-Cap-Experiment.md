# ChatGPT Micro-Cap Experiment: Can AI Trade Stocks Successfully?

[View the original repository on GitHub](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment)

This project explores the potential of large language models like ChatGPT to generate alpha in the stock market, starting with a real-money micro-cap portfolio.

## Project Overview

This experiment uses ChatGPT to manage a real-money micro-cap portfolio. The project seeks to answer whether a powerful AI can make smart trading decisions using real-time data and make profitable trades. The experiment ran from June 2023 to December 2023.

**Key Features:**

*   **AI-Powered Trading:** Utilizes ChatGPT-4 for stock picking and trade execution.
*   **Automated Portfolio Management:** Includes a trading engine with portfolio management and stop-loss automation.
*   **Real-Time Data & Analysis:** Leverages market data to provide the latest performance reports, and generates daily PnL, total equity, and trade history in CSV files.
*   **Performance Tracking:** Visualizes results with Matplotlib graphs comparing ChatGPT's performance against benchmarks.
*   **Comprehensive Documentation:** Provides detailed documentation, methodology, prompts, and Q&A.
*   **Transparent Logging:** Automatically saves logs for complete transparency.
*   **Backtesting Support:** Includes ASOF_DATE override for historical analysis.

## How the Experiment Works

*   **Daily Data:** ChatGPT receives daily trading data on the stocks in its portfolio.
*   **Strict Stop-Loss:** Implements strict stop-loss rules to manage risk.
*   **Weekly Research:** ChatGPT utilizes deep research to re-evaluate its account weekly.
*   **Performance Tracking:** Performance data is tracked and published weekly on my blog: [A.I Controls Stock Account](https://nathanbsmith729.substack.com)

## Repository Structure

*   **`trading_script.py`**: Core trading engine with portfolio management and stop-loss automation.
*   **`Scripts and CSV Files/`**: My personal portfolio (updated every trading day).
*   **`Start Your Own/`**: Template files and guide for starting your own experiment.
*   **`Weekly Deep Research (MD|PDF)/`**: Research summaries and performance reports.
*   **`Experiment Details/`**: Documentation, methodology, prompts, and Q&A.

## Current Performance

![Latest Performance Results](Results.png)

**Current Status:** Portfolio is outperforming the S&P 500 benchmark

*Performance data is updated after each trading day. See the CSV files in `Scripts and CSV Files/` for detailed daily tracking.*

## Technologies Used

*   **Python:** Core scripting and automation.
*   **pandas + yFinance:** Market data fetching and analysis.
*   **Matplotlib:** Performance visualization and charting.
*   **ChatGPT-4:** AI-powered trading decision engine.

## Get Started

*   [Start Your Own](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/blob/main/Start%20Your%20Own/README.md)

## Contributing

This project is open to contributions!

*   **Issues:** Report bugs or suggest improvements.
*   **Pull Requests:** Submit your contributions.
*   **Collaboration:** High-value contributors may be invited as maintainers/admins.

## Tech Stack and Features

**Core Technologies**

*   **Python**: Core scripting and automation
*   **pandas + yFinance**: Market data fetching and analysis
*   **Matplotlib**: Performance visualization and charting
*   **ChatGPT-4**: AI-powered trading decision engine

**Key Features**

*   **Robust Data Sources**: Yahoo Finance primary, Stooq fallback for reliability
*   **Automated Stop-Loss**: Automatic position management with configurable stop-losses
*   **Interactive Trading**: Market-on-Open (MOO) and limit order support
*   **Backtesting Support**: ASOF_DATE override for historical analysis
*   **Performance Analytics**: CAPM analysis, Sharpe/Sortino ratios, drawdown metrics
*   **Trade Logging**: Complete transparency with detailed execution logs

## System Requirements

*   Python 3.11+
*   Internet connection for market data
*   ~10MB storage for CSV data files

## Contact

Have feature requests or any advice?

Please reach out here: **nathanbsmith.business@gmail.com**