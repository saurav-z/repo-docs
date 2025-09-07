# ChatGPT Micro-Cap Experiment: Can AI Beat the Market?

**[Explore the live trading experiment where ChatGPT manages a real-money micro-cap portfolio, and see if AI can generate alpha.](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment)**

## Overview

This project details a six-month live trading experiment utilizing ChatGPT to manage a real-money micro-cap portfolio. The goal is to investigate the potential of Large Language Models (LLMs) like ChatGPT in making intelligent trading decisions using real-time market data.  This repository provides the tools, data, and insights to understand and potentially replicate this experiment.

## Key Features

*   **AI-Powered Trading:** Leveraging ChatGPT for trading decisions.
*   **Real-World Portfolio Management:**  Manages a live micro-cap portfolio with a defined budget.
*   **Automated Trading:** Includes a trading engine with portfolio management and stop-loss automation.
*   **Performance Tracking:** Detailed tracking of P&L, total equity, and trade history.
*   **Transparent Data:** Provides CSV files for each trading day, along with logs for full transparency.
*   **Performance Visualization:** Uses Matplotlib to compare ChatGPT's performance against benchmarks.
*   **Deep Research Reports:** Weekly research summaries and performance reports.
*   **Open Source:** Open to community contributions and collaboration.

## Repository Structure

*   **`trading_script.py`**: The main trading engine with portfolio management and stop-loss automation.
*   **`Scripts and CSV Files/`**:  Contains daily updates on the portfolio performance.
*   **`Start Your Own/`**:  Provides templates and guides to help you start your own AI trading experiment.
*   **`Weekly Deep Research (MD|PDF)/`**:  Contains research summaries and performance reports.
*   **`Experiment Details/`**:  Documentation, methodology, prompts, and Q&A.

## How it Works

Each trading day, ChatGPT receives market data on the stocks in its portfolio. Strict stop-loss rules are applied. Every week, ChatGPT leverages deep research to re-evaluate its account.

## Current Performance

**(Insert Performance Chart Here - e.g., a graph from Results.png)**

**Current Status:** Portfolio is outperforming the S&P 500 benchmark.

*Performance data is updated after each trading day.  See the CSV files in `Scripts and CSV Files/` for detailed daily tracking.*

## Tech Stack

*   **Python**: Core scripting and automation.
*   **pandas + yFinance**: Market data fetching and analysis.
*   **Matplotlib**: Performance visualization and charting.
*   **ChatGPT-4**: AI-powered trading decision engine.

## Core Features

*   **Robust Data Sources**: Yahoo Finance (primary), Stooq (fallback).
*   **Automated Stop-Loss**: Automatic position management with configurable stop-losses.
*   **Interactive Trading**: Market-on-Open (MOO) and limit order support.
*   **Backtesting Support**: ASOF_DATE override for historical analysis.
*   **Performance Analytics**: CAPM analysis, Sharpe/Sortino ratios, drawdown metrics.
*   **Trade Logging**: Complete transparency with detailed execution logs.

## System Requirements

*   Python 3.11+
*   Internet connection for market data
*   ~10MB storage for CSV data files

## Follow Along

The experiment runs from June 2025 to December 2025. Portfolio data is updated daily.  Weekly updates are posted on the author's blog: [A.I Controls Stock Account](https://nathanbsmith729.substack.com)

## Contribute

Contributions are welcome!  Please submit issues or pull requests. High-value contributors may be invited as maintainers.

## Contact

For feature requests or any advice, reach out to: **nathanbsmith.business@gmail.com**