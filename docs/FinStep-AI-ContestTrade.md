<p align="center">
  <img src="assets/logo.jpg" style="width: 100%; height: auto;">
</p>
<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/abs/2508.00554" target="_blank"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2508.00554-B31B1B?logo=arxiv"/></a>
  <a href="https://opensource.org/licenses/Apache-2.0" target="_blank"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"/></a>
  <a href="https://www.python.org/downloads/release/python-3100/" target="_blank"><img alt="Python Version" src="https://img.shields.io/badge/Python-3.10+-brightgreen.svg"/></a>
  <a href="https://www.reddit.com/r/ContestTradeCommunity/?feed=home" target="_blank"><img alt="Reddit" src="https://img.shields.io/badge/Reddit-ContestTradeCommunity-orange?logo=reddit&logoColor=white"/></a>
  <a href="./assets/wechat.png" target="_blank"><img alt="WeChat" src="https://img.shields.io/badge/WeChat-ContestTrade-brightgreen?logo=wechat&logoColor=white"/></a>
</div>
<div align="center">
  <a href="README.md">English</a> | <a href="README_cn.md">中文</a>
</div>

---
# ContestTrade: Your AI-Powered Event-Driven Trading Framework

**[ContestTrade](https://github.com/FinStep-AI/ContestTrade)** is an innovative multi-agent trading framework designed to autonomously identify and capitalize on event-driven investment opportunities.

## Key Features

*   **Automated Stock Selection:**  Scans the entire market to automatically generate tradable stock lists, eliminating the need for manual screening.
*   **Event-Driven Strategy:** Focuses on catalysts like news, announcements, capital flows, and policy changes to pinpoint high-impact opportunities.
*   **Customizable Configuration:**  Supports user-defined agent research preferences and strategies for flexible adaptation to various investment styles.
*   **Multi-Agent Architecture:** Employs a two-stage pipeline to simulate the decision-making process of an investment firm, ensuring robust and effective insights.

## Introduction

ContestTrade is a multi-agent trading framework specializing in event-driven stock selection.  The system aims to automatically discover, evaluate, and track event-driven opportunities with investment value without human intervention, and ultimately provide executable asset allocation recommendations.

## Framework Overview

<p align="center">
  <img src="assets/architecture.jpg" style="width: 90%; height: auto;">
</p>

ContestTrade utilizes a structured two-stage pipeline:

1.  **Data Processing Stage:**  Market data is fed into the **Data Team**, where multiple Data Analysis Agents refine raw data into structured "textual factors." An internal contest mechanism evaluates the factors, creating an optimal "factor portfolio."
2.  **Research and Decision-Making Stage:** The factor portfolio is then passed to the **Research Team**. Multiple Research Agents analyze these factors and submit trading proposals. A second internal contest synthesizes these proposals into a unified asset allocation strategy.

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/FinStep-AI/ContestTrade.git
cd ContestTrade

# 2. (Recommended) Create and activate a virtual environment
conda create -n contesttrade python=3.10
conda activate contesttrade

# 3. Install dependencies
pip install -r requirements.txt
```

Or deploy with [Docker](https://docs.n8n.io/hosting/installation/docker/):

```
docker run -it --rm --name contest_trade -v $(pwd)/config.yaml:/ContestTrade/config.yaml finstep/contesttrade:v2.0
```

## Configuration

Configure API keys and LLM parameters in `config_us.yaml`:

| Key                      | Description                                | Required |
| :----------------------- | :----------------------------------------- | :------- |
| `TUSHARE_KEY`            | Tushare data interface key                 | No       |
| `BOCHA_KEY`              | Bocha search engine key                    | No       |
| `SERP_KEY`               | SerpAPI search engine key                  | No       |
| `FMP_KEY`                | FMP API key                                | **Yes**  |
| `FINNHUB_KEY`            | Finnhub API key                            | No       |
| `ALPHA_VANTAGE_KEY`      | Alpha Vantage API key                      | **Yes** |
| `POLYGON_KEY`            | Polygon API key                            | **Yes** |
| `LLM_API_KEY`            | LLM API key for general tasks              | **Yes** |
| `LLM_BASE_URL`           | LLM API base URL for general tasks         | **Yes** |
| `LLM_THINKING_API_KEY`   | LLM API key for complex reasoning          | No       |
| `LLM_THINKING_BASE_URL`  | LLM API base URL for complex reasoning     | No       |
| `VLM_API_KEY`            | VLM API key for visual analysis            | No       |
| `VLM_BASE_URL`           | VLM API base URL for visual analysis       | No       |

> Note:  You must obtain your own LLM and VLM API keys, specifying the URL, API Key, and model name according to your chosen platform.

## Preference Settings

Customize Research Agent "trading beliefs" in `contest_trade/config/belief_list.json`:

**Example 1: Short-Term Event-Driven (Aggressive)**
```json
[
  "Focus on short-term event-driven opportunities: prioritize company announcements, M&A and restructuring, sudden order increases, technological breakthroughs and other catalysts; prefer mid/small-cap, high-volatility thematic stocks, suitable for aggressive arbitrage strategies."
]
```

**Example 2: Stable Events (Conservative)**
```json
[
  "Focus on stable, high-certainty events: pay attention to dividends, buybacks, earnings forecast confirmations, major contract signings and policy tailwinds; prefer large-cap blue-chips with low volatility and high certainty, suitable for conservative allocation."
]
```

## Usage

Run ContestTrade via the Command Line Interface (CLI):

```bash
python -m cli.main run
```

Follow the interactive terminal interface to select a market and view the results.

<p align="center">
  <img src="assets/contest_trade_cli_select_market.jpg" style="width: 100%; height: auto;">
</p>

<p align="center">
  <img src="assets/contest_trade_cli_main_us.jpg" style="width: 100%; height: auto;">
</p>

<p align="center">
  <img src="assets/contest_trade_cli_report_us.jpg" style="width: 100%; height: auto;">
</p>

<p align="center">
  <img src="assets/contest_trade_cli_research_report_us.jpg" style="width: 100%; height: auto;">
</p>

<p align="center">
  <img src="assets/contest_trade_cli_data_report_us.jpg" style="width: 100%; height: auto;">
</p>

> Reports are saved in Markdown format in the `contest_trade/agents_workspace/results` directory.

## 🌟 Vision & Roadmap

The project aims to leverage the power of the open-source community to explore new paradigms of quantitative trading in the AGI era, creating a stable, trustworthy, and scalable agent trading framework.

### Roadmap

**V1.1 (Finished):** Framework Stability Enhancement & Core Experience Optimization

**V2.0 (Finished):** Market and Function Expansion
- [✓] Access to **US stock** market data.
- [✓] Introduced richer factors and signal sources.

**Future Plans:**
- [ ] Support for Hong Kong stocks and other markets
- [ ] Visual backtesting and analysis interface
- [ ] Support for scaling up more agents

## Contributing

Contribute to ContestTrade!  See the **[Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md)** for developer contributions.  Non-code contributions, such as feature suggestions or bug reports, are also welcome on the [Issues page](https://github.com/FinStep-AI/ContestTrade/issues).

## Star History

<div align="center">
  <a href="https://star-history.com/#FinStep-AI/ContestTrade&Date">
    <img src="https://api.star-history.com/svg?repos=FinStep-AI/ContestTrade&type=Date" alt="Star History Chart" style="width: 80%;">
  </a>
</div>

## Disclaimer

**Important Disclaimer:** This project, `ContestTrade`, is an open-source quantitative trading research framework intended for academic and educational purposes only. The examples, data, and analysis results included in the project do not constitute any form of investment advice.

**Risk Warning:**
*   **Market Risk:** This project does not constitute any form of investment, financial, legal, or tax advice. All outputs, including trading signals and analyses, are the results of AI model deductions based on historical data and should not be considered a basis for any buy or sell operations.
*   **Data Accuracy:** The data sources used by the framework may be subject to delays, inaccuracies, or incompleteness. We do not guarantee the reliability of the data.
*   **Model Hallucination:** AI models (including Large Language Models) have inherent limitations and are subject to the risk of "hallucination." We do not guarantee the accuracy, completeness, or timeliness of the information generated by the framework.
*   **Liability:** The developers assume no liability for any direct or indirect losses resulting from the use of or inability to use this framework. Investing involves risks; enter the market with caution.

**Before using this framework for any actual trading decisions, be sure to fully understand the associated risks.**

## Citation

```bibtex
@misc{zhao2025contesttrade,
      title={ContestTrade: A Multi-Agent Trading System Based on Internal Contest Mechanism}, 
      author={Li Zhao and Rui Sun and Zuoyou Jiang and Bo Yang and Yuxiao Bai and Mengting Chen and Xinyang Wang and Jing Li and Zuo Bai},
      year={2025},
      eprint={2508.00554},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR}
}
```

## License

This project is licensed under the [Apache 2.0 License](LICENSE).