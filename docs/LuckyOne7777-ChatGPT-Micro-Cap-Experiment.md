# ChatGPT Micro-Cap Experiment: Can AI Beat the Market?

**Can a large language model like ChatGPT make profitable trading decisions in the micro-cap stock market?** This project explores that question through a live trading experiment.

[View the original repository on GitHub](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment)

## Project Overview

This repository documents a six-month, real-money trading experiment where ChatGPT manages a portfolio of micro-cap stocks. The goal is to determine if an AI, using real-time market data and with minimal human intervention, can generate alpha.

## Key Features

*   **AI-Driven Trading:** ChatGPT-4 is the primary decision-maker, evaluating market data and making trade recommendations.
*   **Real-Time Data & Analysis:** Utilizes market data feeds to track portfolio performance daily.
*   **Automated Stop-Loss:** Implements strict stop-loss rules for risk management.
*   **Performance Tracking:** Detailed performance data, including P&L, equity, and trade history, is tracked in CSV files.
*   **Transparency:** Comprehensive logs, including trade executions, research summaries, and ChatGPT prompts, are available for review.
*   **Regular Updates:** Portfolio performance data and weekly analysis are published.

## Repository Structure

*   **`trading_script.py`:** The core trading engine, managing the portfolio and stop-loss automation.
*   **`Scripts and CSV Files/`:** Contains the experiment's performance data, updated daily.
*   **`Start Your Own/`:** Provides templates and a guide to replicate the experiment.
*   **`Weekly Deep Research (MD|PDF)/`:** Includes research summaries and performance reports.
*   **`Experiment Details/`:** Contains documentation, methodology, prompts, and Q&A.

## Tech Stack

*   **Python:** The primary programming language for scripting and automation.
*   **pandas + yFinance:** Used for market data retrieval and analysis.
*   **Matplotlib:** Visualizes portfolio performance data.
*   **ChatGPT-4:** The AI-powered trading decision engine.

## Current Performance

![Latest Performance Results](Results.png)

**Current Status:** Portfolio is outperforming the S&P 500 benchmark

*Performance data is updated after each trading day. See the CSV files in `Scripts and CSV Files/` for detailed daily tracking.*

## How to Get Started

Learn how to set up your own experiment with the provided templates and guide:  [Starting Your Own](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment/blob/main/Start%20Your%20Own/README.md)

## Why This Matters

This project explores the potential of AI in financial markets. By analyzing ChatGPT's trading decisions, we can gain valuable insights into the future of algorithmic trading and the capabilities of large language models.

## Contribute

Your contributions are welcome!

*   **Issues:** Report any bugs or suggest improvements.
*   **Pull Requests:** Submit your changes for review.

## Stay Updated

*   **Blog:**  [A.I Controls Stock Account](https://nathanbsmith729.substack.com)
*   **Email:**  nathanbsmith.business@gmail.com (for feature requests or advice)