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

# ContestTrade: Automated AI-Driven Trading Framework

**ContestTrade is a groundbreaking multi-agent trading system that autonomously identifies and capitalizes on event-driven investment opportunities, offering a new frontier in algorithmic trading.**  [Explore the original repository on GitHub](https://github.com/FinStep-AI/ContestTrade).

## Key Features

*   **Automated Stock Selection:** Scans the entire market to generate tradable stock lists without manual screening.
*   **Event-Driven Strategy:** Focuses on catalysts like news, announcements, and capital flows for high-impact opportunities.
*   **Customizable Configuration:** Supports user-defined agent preferences and trading strategies for flexible adaptation.
*   **Multi-Agent Architecture:** Employs a two-stage pipeline with data and research teams, each using internal contest mechanisms for robust decision-making.
*   **Supports US Market Data**: Access to US stock market data.
*   **CLI Interface**: Allows users to interact with the system through a command-line interface.

## Introduction

ContestTrade is a cutting-edge multi-agent trading framework designed for event-driven stock selection. The system autonomously discovers, evaluates, and tracks event-driven opportunities with potential investment value, culminating in executable asset allocation recommendations.

## Framework Overview

<p align="center">
  <img src="assets/architecture.jpg" style="width: 90%; height: auto;">
</p>

The ContestTrade system utilizes a structured two-stage pipeline, mirroring the decision-making process of an investment firm:

1.  **Data Processing Stage:** The **Data Team** processes raw market data from multiple sources. Data Analysis Agents refine this into structured "textual factors," and an internal contest mechanism selects the optimal "factor portfolio."
2.  **Research and Decision-Making Stage:** The **Research Team** receives the optimal factor portfolio. Research Agents, each with unique "Trading Beliefs" and access to financial tools, analyze factors and submit trading proposals. A second internal contest synthesizes a unified asset allocation strategy.

## Installation

```bash
# 1. Clone the project repository
git clone https://github.com/FinStep-AI/ContestTrade.git
cd ContestTrade

# 2. (Recommended) Create and activate a virtual environment
conda create -n contesttrade python=3.10
conda activate contesttrade

# 3. Install project dependencies
pip install -r requirements.txt
```

Or deploy with [Docker](https://docs.n8n.io/hosting/installation/docker/):

```
docker run -it --rm --name contest_trade -v $(pwd)/config.yaml:/ContestTrade/config.yaml finstep/contesttrade:v1.1
```

## Configuration

Before running ContestTrade, configure API keys and LLM parameters in `config_us.yaml`.

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

> Note: Obtain LLM and VLM API keys and fill in the required URLs, API Keys, and model names.

## Preference

Each Research Agent operates based on a "trading belief" defined in `contest_trade/config/belief_list.json`.

Example Preference (short-term, aggressive):
```json
[
  "Focus on short-term event-driven opportunities: prioritize company announcements, M&A and restructuring, sudden order increases, technological breakthroughs and other catalysts; prefer mid/small-cap, high-volatility thematic stocks, suitable for aggressive arbitrage strategies."
]
```

Example Preference (stable, conservative):
```json
[
  "Focus on stable, high-certainty events: pay attention to dividends, buybacks, earnings forecast confirmations, major contract signings and policy tailwinds; prefer large-cap blue-chips with low volatility and high certainty, suitable for conservative allocation."
]
```

## Usage

Use the Command Line Interface (CLI):

```bash
python -m cli.main run
```

The CLI provides an interactive terminal interface to select a market and analyze data.
<p align="center">
  <img src="assets/contest_trade_cli_select_market.jpg" style="width: 100%; height: auto;">
</p>

View agent signals in the results summary.
<p align="center">
  <img src="assets/contest_trade_cli_main_us.jpg" style="width: 100%; height: auto;">
</p>

Access detailed research reports.
<p align="center">
  <img src="assets/contest_trade_cli_report_us.jpg" style="width: 100%; height: auto;">
</p>
<p align="center">
  <img src="assets/contest_trade_cli_research_report_us.jpg" style="width: 100%; height: auto;">
</p>
View detailed data analysis reports.
<p align="center">
  <img src="assets/contest_trade_cli_data_report_us.jpg" style="width: 100%; height: auto;">
</p>

> All reports are saved in Markdown format in the `contest_trade/agents_workspace/results` directory.

## 🌟 Vision & Roadmap

We aim to leverage open-source collaboration to advance quantitative trading in the AGI era.

### Roadmap

**V1.1 (Completed): Framework Stability & Core Experience Optimization**
-   [✓] Decoupled core data source module.
-   [✓] Improved CLI logging and interaction.

**V2.0 (Completed): Market and Function Expansion**
-   [✓] US Stock Market Data integration.
-   [✓] Expanded factors and signal sources.

**Future Plans:**
-   [ ] Support for Hong Kong and other markets.
-   [ ] Visual backtesting and analysis interface.
-   [ ] Support for more agents.

## Contributing

Contribute to ContestTrade! See the **[Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md)**.

We welcome feature suggestions, bug reports ([Go to Issues page](https://github.com/FinStep-AI/ContestTrade/issues)), and feedback.

## Star History

<div align="center">
  <a href="https://star-history.com/#FinStep-AI/ContestTrade&Date">
    <img src="https://api.star-history.com/svg?repos=FinStep-AI/ContestTrade&type=Date" alt="Star History Chart" style="width: 80%;">
  </a>
</div>

## Disclaimer

**Important Disclaimer:** ContestTrade is for academic and educational purposes only. No investment advice is provided.

**Risk Warning:**
*   **Market Risk:** This project is not financial, legal, or tax advice. AI-generated outputs are based on historical data and should not be used for investment decisions.
*   **Data Accuracy:** Data sources may have delays, inaccuracies, or be incomplete.
*   **Model Hallucination:** AI models have limitations and can "hallucinate".
*   **Liability:** Developers are not liable for any losses from using this framework. Investing involves risks.

**Understand the risks before using this framework for trading.**

## Citation

Cite our paper:

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

Licensed under the [Apache 2.0 License](LICENSE).