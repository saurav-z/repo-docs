# ChatGPT Micro-Cap Trading Experiment: Can AI Beat the Market?

**Can a large language model like ChatGPT actually make smart trading decisions and generate alpha in the stock market?** This repository details a real-world, six-month experiment where ChatGPT manages a live micro-cap portfolio.

[View the original repository on GitHub](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment)

## Key Features:

*   **AI-Powered Trading:** Leverages ChatGPT-4 to make real-time trading decisions.
*   **Live Portfolio Tracking:** Tracks a real-money micro-cap portfolio, updated daily.
*   **Performance Analysis:** Detailed performance metrics, including P&L, equity, and trade history, with comparisons to market benchmarks.
*   **Automated Stop-Loss:** Implements strict stop-loss rules for risk management.
*   **Transparency:** Provides complete trading data, research summaries, and methodology documentation.
*   **Open Source:** Use the project as a blueprint for your own AI trading experiments.

## Project Overview

This project is a bold experiment to assess the potential of AI in financial markets. Using an initial investment of just $100, the project aims to determine if ChatGPT can generate alpha (or at least make smart trading decisions) using real-time data and strict risk management principles.

### Daily Trading Process:

1.  **Data Input:** Provides ChatGPT with up-to-date trading data on portfolio stocks.
2.  **Decision Making:** ChatGPT analyzes data and makes trading recommendations.
3.  **Execution:** Trades are executed with stop-loss orders in place.
4.  **Weekly Research:** ChatGPT uses deep research to reevaluate its account.
5.  **Performance Tracking:** Performance data is tracked and published weekly.

## Repository Structure

*   **`trading_script.py`**: Main trading engine for portfolio management and stop-loss automation.
*   **`Scripts and CSV Files/`**:  Contains the portfolio and trade data, updated daily.
*   **`Start Your Own/`**: Template files and guides for starting your own AI trading experiment.
*   **`Weekly Deep Research (MD|PDF)/`**:  Summaries and performance reports, allowing deep dives into the reasoning behind trades.
*   **`Experiment Details/`**: Detailed documentation, methodology, prompts, and a Q&A section.

## Current Performance

<!-- To update performance chart: 
     1. Replace the image file with updated results
     2. Update the dates and description below
     3. Update the "Last Updated" date -->

**Last Updated:** August 2025

![Latest Performance Results](%286-30%20-%208-15%29%20Results.png)

**Current Status:** Portfolio is outperforming the S&P 500 benchmark

*Performance data is updated after each trading day. See the CSV files in `Scripts and CSV Files/` for detailed daily tracking.*

## Technical Details

### Core Technologies:

*   **Python:** Core scripting and automation.
*   **pandas + yFinance:** Market data fetching and analysis.
*   **Matplotlib:** Performance visualization and charting.
*   **ChatGPT-4:** AI-powered trading decision engine.

### Key Features & Functionality:

*   **Robust Data Sources:** Utilizes Yahoo Finance for primary data, with Stooq as a fallback for reliability.
*   **Automated Stop-Loss:** Includes automatic position management with configurable stop-losses to minimize risk.
*   **Interactive Trading:** Supports Market-on-Open (MOO) and limit orders for flexible trading.
*   **Backtesting Support:**  Includes an ASOF_DATE override for historical analysis.
*   **Performance Analytics:** Implements CAPM analysis, Sharpe/Sortino ratios, and drawdown metrics.
*   **Trade Logging:** Provides complete transparency with detailed execution logs.

### System Requirements:

*   Python 3.7+
*   Internet connection for market data.
*   Approximately 10MB of storage for CSV data files.

## Follow the Experiment

The experiment runs from June 2025 to December 2025.  Portfolio CSV files are updated daily.

Updates are posted weekly on the author's blog: [Link to Blog](https://nathanbsmith729.substack.com).

**Contact:** nathanbsmith.business@gmail.com