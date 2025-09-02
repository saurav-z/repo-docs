# ChatGPT Micro-Cap Experiment: Can AI Outperform the Market?

**Explore a live trading experiment where ChatGPT manages a micro-cap portfolio with real money, aiming to generate alpha through AI-driven stock picks.**  [View the original repository here.](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment)

## Project Overview

This project documents a six-month live trading experiment designed to assess the effectiveness of large language models, specifically ChatGPT, in making real-time trading decisions within the micro-cap stock market. Using a real-money portfolio, this experiment aims to answer the critical question: Can AI generate alpha (or at least make smart trading decisions) using real-time data?

## Key Features

*   **AI-Powered Trading:** Leverages ChatGPT-4 to generate stock trading signals.
*   **Real-Time Data Integration:** Uses live market data for informed decision-making.
*   **Automated Portfolio Management:** Includes stop-loss rules, and daily trading updates.
*   **Comprehensive Performance Tracking:** Detailed daily performance data, including P&L, total equity, and trade history, are tracked.
*   **Transparent Reporting:** Weekly performance reports and research summaries.
*   **Open-Source & Replicable:** Includes scripts, prompts, and templates to allow others to start their own experiments.

## Repository Structure

*   **`trading_script.py`:** The core trading engine that manages the portfolio and implements stop-loss automation.
*   **`Scripts and CSV Files/`:** Contains the experiment's portfolio data, updated daily.
*   **`Start Your Own/`:** Provides template files and guidance for replicating the experiment.
*   **`Weekly Deep Research (MD|PDF)/`:** Contains research summaries and performance reports.
*   **`Experiment Details/`:** Includes documentation, the methodology, prompts used, and a Q&A section.

## How It Works

Each trading day, the AI is provided with trading data for the stocks in its portfolio. Strict stop-loss rules are applied to manage risk. The AI reevaluates its account using deep research on a weekly basis.

## Current Performance

**Last Updated:** August 29th, 2025

![Latest Performance Results](Results.png)

**Current Status:** Portfolio is outperforming the S&P 500 benchmark.

*Performance data is updated after each trading day. See the CSV files in `Scripts and CSV Files/` for detailed daily tracking.*

## Why This Matters

As AI becomes increasingly integrated across industries, this project provides valuable insights into the potential of AI-driven financial management. It offers a transparent, data-driven exploration of AI's capabilities in the financial market.

## Technology Stack

**Core Technologies:**

*   **Python:** For core scripting and automation.
*   **pandas + yFinance:** For market data fetching and analysis.
*   **Matplotlib:** For performance visualization and charting.
*   **ChatGPT-4:** The AI-powered trading decision engine.

**Key Features:**

*   **Robust Data Sources:** Uses Yahoo Finance as the primary data source with Stooq as a fallback.
*   **Automated Stop-Loss:** Automatically manages positions with configurable stop-loss.
*   **Interactive Trading:** Supports Market-on-Open (MOO) and limit orders.
*   **Backtesting Support:** Provides an ASOF\_DATE override for historical analysis.
*   **Performance Analytics:** Includes CAPM analysis, Sharpe/Sortino ratios, and drawdown metrics.
*   **Trade Logging:** Maintains complete transparency with detailed execution logs.

## Getting Started

Follow the experiment's progress on the author's blog for weekly updates: [A.I Controls Stock Account](https://nathanbsmith729.substack.com)

If you feel inspired to do something similar, feel free to use this as a blueprint.

**[View the original repository here.](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment)**

**Contact:**
For feature requests or advice, please contact nathanbsmith.business@gmail.com.