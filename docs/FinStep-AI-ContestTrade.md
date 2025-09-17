<p align="center">
  <img src="assets/logo.jpg" style="width: 100%; height: auto;">
</p>

<div align="center">
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

# ContestTrade: Revolutionizing AI-Driven Trading with a Multi-Agent System

**ContestTrade** is a cutting-edge, multi-agent trading framework designed to automatically identify and capitalize on event-driven investment opportunities.  Dive deeper into the project on [GitHub](https://github.com/FinStep-AI/ContestTrade).

## Key Features

*   **Automated Stock Selection:**  Scans the entire market to automatically generate tradable stock lists, eliminating manual screening.
*   **Event-Driven Strategy:** Focuses on catalysts like news, announcements, and capital flows to identify high-impact opportunities.
*   **Personalized Configuration:**  Allows customization of agent research preferences and strategies to adapt to diverse investment styles.
*   **Multi-Agent Architecture:**  Employs a two-stage pipeline mimicking investment firm decision-making, ensuring robust and effective insights.

## Introduction

ContestTrade offers a groundbreaking multi-agent framework specifically engineered for event-driven stock selection. It autonomously discovers, evaluates, and tracks promising investment opportunities, ultimately providing actionable asset allocation recommendations without human intervention.

## Framework Overview

<p align="center">
  <img src="assets/architecture.jpg" style="width: 90%; height: auto;">
</p>

The system operates via a two-stage process, mirroring the decision-making process of an investment firm, ensuring final investment decisions are based on reliable insights.

1.  **Data Processing Stage:** The **Data Team** receives raw market data from numerous sources, transforming it into structured "textual factors." An internal contest mechanism assesses the value of each data agent's factors to create an optimal "factor portfolio."

2.  **Research and Decision-Making Stage:** The **Research Team** receives the portfolio, with individual Research Agents each using unique "Trading Beliefs" and a suite of financial tools to conduct detailed analyses and propose trading plans. A second contest evaluates these proposals to generate a unified asset allocation strategy.

## Installation

Get started with ContestTrade by following these steps:

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

Alternatively, deploy with Docker:

```
docker run -it --rm --name contest_trade -v $(pwd)/config.yaml:/ContestTrade/config.yaml finstep/contesttrade:v2.0
```

## Configuration

Before using ContestTrade, configure API keys and LLM parameters in `config_us.yaml`:

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

>  *Note:* You need to get your own LLM and VLM API keys and fill in the corresponding URLs.

## Preference: Defining Trading Beliefs

Customize your investment strategy by configuring "trading beliefs" for each Research Agent in `contest_trade/config/belief_list.json`.

**Examples:**

*   **Short-Term, Aggressive:** `"Focus on short-term event-driven opportunities... suitable for aggressive arbitrage strategies."`
*   **Stable, Conservative:** `"Focus on stable, high-certainty events... suitable for conservative allocation."`

**Default Configuration:**
```json
[
  "Based on the provided information, comprehensively consider each company's business dynamics, industry trends and potential market impact. Recommend stock portfolios with short-term investment potential for two groups: Group 1 — risk-seekers (prefer high volatility, high returns, mid/low market cap stocks); Group 2 — conservative investors (prefer low volatility, stable returns, large-cap stocks).",
  "Based on recent sudden events, policy adjustments and company announcements as catalysts, combined with market sentiment transmission paths and capital game features. Screen event-driven opportunities for two different styles: Group 1 — aggressive arbitrage (prefer restructuring expectation, sudden order increases, technical breakthroughs in small-cap stocks); Group 2 — defensive arbitrage (prefer dividend increases, large buybacks, acquisition of franchise rights in blue-chip stocks). Pay attention to northbound capital movement and institutional seat trends on the trading leaderboard for resonance effects."
]
```

## Usage: Running ContestTrade

Use the Command Line Interface (CLI) to launch ContestTrade:

```bash
python -m cli.main run
```

<p align="center">
  <img src="assets/contest_trade_cli_select_market.jpg" style="width: 100%; height: auto;">
</p>

View agent signals after completion:
<p align="center">
  <img src="assets/contest_trade_cli_main_us.jpg" style="width: 100%; height: auto;">
</p>

Access detailed reports:
<p align="center">
  <img src="assets/contest_trade_cli_report_us.jpg" style="width: 100%; height: auto;">
</p>

<p align="center">
  <img src="assets/contest_trade_cli_research_report_us.jpg" style="width: 100%; height: auto;">
</p>

<p align="center">
  <img src="assets/contest_trade_cli_data_report_us.jpg" style="width: 100%; height: auto;">
</p>

>  *Note:* Reports are saved in the `contest_trade/agents_workspace/results` directory in Markdown format.

## 🌟 Vision & Roadmap

Our goal is to use open-source collaboration to explore the future of quantitative trading in the AGI era.

### Roadmap

*   **V1.1 (Finished):** Framework Stability Enhancement & Core Experience Optimization
    *   Decoupled data source module.
    *   Optimized CLI logging.
*   **V2.0 (Finished):** Market and Function Expansion
    *   US stock market data access.
    *   Expanded factors and signal sources.

*   **Future Plans:**
    *   Support for Hong Kong and other markets.
    *   Visual backtesting and analysis interface.
    *   Support for scaling up more agents.

## Contributing

Contribute to ContestTrade!  See the **[Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md)** for development guidelines.

We also welcome non-code contributions:
*   [Submit Feature Suggestions/Bug Reports](https://github.com/FinStep-AI/ContestTrade/issues)
*   Provide test results feedback.

## Star History

<div align="center">
  <a href="https://star-history.com/#FinStep-AI/ContestTrade&Date">
    <img src="https://api.star-history.com/svg?repos=FinStep-AI/ContestTrade&type=Date" alt="Star History Chart" style="width: 80%;">
  </a>
</div>

## Disclaimer

**Important:** ContestTrade is for research and educational purposes only and does *not* offer investment advice.

**Risk Warning:**
*   **Market Risk:** This project is for informational purposes and is *not* financial advice.  AI-generated outputs are based on historical data and should not be the basis for trading decisions.
*   **Data Accuracy:** Data may be delayed, inaccurate, or incomplete.
*   **Model Hallucination:** AI models have limitations; we do not guarantee accuracy.
*   **Liability:** Developers are not liable for any losses.  Investing involves risk.

**Thoroughly understand the risks before using this framework for any trading.**

## Citation

If you use ContestTrade in your research, please cite our paper:

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