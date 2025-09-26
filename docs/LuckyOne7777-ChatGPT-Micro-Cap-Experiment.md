# ChatGPT Micro-Cap Experiment: Can AI Trade Stocks?

**Explore the cutting-edge experiment where ChatGPT manages a real-money micro-cap portfolio, available on GitHub!**  ([View Original Repo](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment))

This project investigates the potential of large language models like ChatGPT to make smart trading decisions using real-time data. This is a live experiment, running for six months, where ChatGPT makes the calls.

## Key Features

*   **AI-Powered Trading:** ChatGPT drives the trading decisions.
*   **Real-Money Portfolio:** A live micro-cap portfolio is managed using ChatGPT.
*   **Daily Performance Tracking:** Detailed daily performance data, including P&L, total equity, and trade history, stored in CSV files.
*   **Automated Stop-Loss:**  Strict stop-loss rules are automatically applied.
*   **Weekly Research Integration:** Weekly deep research summaries are integrated to inform decisions.
*   **Performance Visualization:** Matplotlib graphs compare ChatGPT's performance against benchmarks.
*   **Transparent Logging:** Detailed logs and trade data are saved automatically.
*   **Community Driven:** Contributions are welcome!

## Repository Structure

*   **`trading_script.py`:** The core trading engine with portfolio management and stop-loss automation.
*   **`Scripts and CSV Files/`:** Contains the daily updated portfolio data.
*   **`Start Your Own/`:** Provides template files and a guide to starting your own experiment.
*   **`Weekly Deep Research (MD|PDF)/`:** Contains research summaries and performance reports.
*   **`Experiment Details/`:** Includes documentation, methodology, prompts, and Q&A.

## Current Performance

*   **Portfolio Underperforming S&P 500 Benchmark**
*   *(Performance data is updated after each trading day. See the CSV files in `Scripts and CSV Files/` for detailed daily tracking.)*

## Tech Stack & Features

**Core Technologies:**

*   **Python:** Core scripting and automation
*   **pandas + yFinance:** Market data fetching and analysis
*   **Matplotlib:** Performance visualization and charting
*   **ChatGPT-4:** AI-powered trading decision engine

**Key Features:**

*   Robust Data Sources (Yahoo Finance, Stooq fallback)
*   Automated Stop-Loss
*   Interactive Trading (Market-on-Open (MOO) and limit order support)
*   Backtesting Support (ASOF\_DATE override)
*   Performance Analytics (CAPM analysis, Sharpe/Sortino ratios, drawdown metrics)
*   Trade Logging

## Getting Started

Dive into the experiment by exploring the detailed documentation in the repo.

*   **To start your own experiment:** [Click here](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/blob/main/Start%20Your%20Own/README.md)
*   **Research Index:** [Deep Research Index](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/blob/main/Experiment%20Details/Deep%20Research%20Index.md)
*   **Disclaimer:** [Disclaimer](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/blob/main/Experiment%20Details/Disclaimer.md)
*   **Q&A:** [Q&A](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/blob/main/Experiment%20Details/Q%26A.md)
*   **Prompts:** [Prompts](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/blob/main/Experiment%20Details/Prompts.md)
*   **Research Summaries (MD):** [Research Summaries (MD)](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/tree/main/Weekly%20Deep%20Research%20(MD))
*   **Full Deep Research Reports (PDF):** [Full Deep Research Reports (PDF)](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/tree/main/Weekly%20Deep%20Research%20(PDF))
*   **Chats:** [Chats](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/blob/main/Experiment%20Details/Chats.md)
*   **Contributing Guide:** [Contributing Guide](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/blob/main/Other/CONTRIBUTING.md)

## System Requirements

*   Python 3.11+
*   Internet connection
*   ~10MB storage for CSV data files

## Follow the Experiment

*   **Blog:** Stay up-to-date with weekly updates and insights on the blog: [A.I Controls Stock Account](https://nathanbsmith729.substack.com)

## Contribute

Contributions are welcome! If you have ideas for improvements or find bugs, please open an issue or submit a pull request.

## Contact

For feature requests, advice, or any questions, reach out: **nathanbsmith.business@gmail.com**