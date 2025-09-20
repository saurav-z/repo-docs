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

# ContestTrade: AI-Powered Multi-Agent Trading Framework

**ContestTrade** is a cutting-edge multi-agent trading framework that leverages AI to automatically identify and capitalize on event-driven investment opportunities, offering a powerful solution for building your own AI trading team.  [Explore the original repo on GitHub](https://github.com/FinStep-AI/ContestTrade)

## Key Features

*   **Automated Stock Selection:**  Scans the entire market and generates tradable stock lists without manual screening.
*   **Event-Driven Focus:**  Identifies opportunities triggered by catalysts like news, announcements, and policy changes.
*   **Customizable Agents:** Supports user-defined research preferences and strategies for flexible adaptation.
*   **Multi-Stage Contest Mechanism:** Employs a two-stage pipeline to ensure robust and reliable investment insights.
*   **Market Coverage:** Supports both US and other markets.

## Introduction

ContestTrade is a multi-agent trading framework designed for event-driven stock selection, aiming to autonomously discover, evaluate, and track valuable investment opportunities. It provides executable asset allocation recommendations without human intervention.

## Framework Overview

ContestTrade's workflow mirrors an investment firm's decision-making process through a structured two-stage pipeline:

1.  **Data Processing Stage:** The **Data Team** receives market data from various sources. Data Analysis Agents transform raw data into structured "textual factors," evaluated through an internal contest to build an optimal "factor portfolio."
2.  **Research and Decision-Making Stage:**  The **Research Team** receives the optimal factor portfolio. Research Agents, each with their "Trading Beliefs" and financial tools, conduct in-depth analyses and submit trading proposals. Another internal contest synthesizes these proposals into a unified asset allocation strategy.

<p align="center">
  <img src="assets/architecture.jpg" style="width: 90%; height: auto;">
</p>

## Installation

Follow these steps to get started:

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

Alternatively, deploy using Docker:

```bash
docker run -it --rm --name contest_trade -v $(pwd)/config.yaml:/ContestTrade/config.yaml finstep/contesttrade:v2.0
```

## Configuration

Configure the necessary API keys and LLM parameters in the `config_us.yaml` file. Here's a table of required and optional keys:

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

> **Note:** You must obtain LLM and VLM API keys, providing the platform, model, URL, and key.

## Preference Configuration

Customize Research Agent "trading beliefs" in the `contest_trade/config/belief_list.json` file.  Each agent can generate up to 5 signals.

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

**Default Configuration:**

```json
[
  "Based on the provided information, comprehensively consider each company's business dynamics, industry trends and potential market impact. Recommend stock portfolios with short-term investment potential for two groups: Group 1 — risk-seekers (prefer high volatility, high returns, mid/low market cap stocks); Group 2 — conservative investors (prefer low volatility, stable returns, large-cap stocks).",
  "Based on recent sudden events, policy adjustments and company announcements as catalysts, combined with market sentiment transmission paths and capital game features. Screen event-driven opportunities for two different styles: Group 1 — aggressive arbitrage (prefer restructuring expectation, sudden order increases, technical breakthroughs in small-cap stocks); Group 2 — defensive arbitrage (prefer dividend increases, large buybacks, acquisition of franchise rights in blue-chip stocks). Pay attention to northbound capital movement and institutional seat trends on the trading leaderboard for resonance effects."
]
```

## Usage

Run ContestTrade using the Command Line Interface (CLI):

```bash
python -m cli.main run
```

The CLI will guide you through market selection and report generation.

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

> Results are saved in the `contest_trade/agents_workspace/results` directory in Markdown format.

## 🌟 Vision & Roadmap

ContestTrade is dedicated to advancing AI in quantitative trading.

### Roadmap

**V1.1 (Finished): Framework Stability Enhancement & Core Experience Optimization**
- [✓] The core data source module is decoupled to achieve adaptors for multiple data sources. (`data-provider` refactor)
- [✓] Optimized CLI logging and interaction experience.

**V2.0 (Finished): Market and Function Expansion**
- [✓] Access to **US stock** market data.
- [✓] Introduced richer factors and signal sources.

**Future Plans:**
- [ ] Support for Hong Kong stocks and other markets
- [ ] Visual backtesting and analysis interface
- [ ] Support for scaling up more agents

## Contributing

Contribute to ContestTrade!  See the **[Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md)**.

Non-code contributions are also welcome:
*   **Suggest features/report bugs:** [Go to Issues page](https://github.com/FinStep-AI/ContestTrade/issues)
*   **Provide feedback:** Share your test results and user experience.

## Star History

<div align="center">
  <a href="https://star-history.com/#FinStep-AI/ContestTrade&Date">
    <img src="https://api.star-history.com/svg?repos=FinStep-AI/ContestTrade&type=Date" alt="Star History Chart" style="width: 80%;">
  </a>
</div>

## Disclaimer

**Important Disclaimer:** This project is an open-source research framework and is for academic and educational purposes only. The content is not financial advice.

**Risk Warning:**
*   **Market Risk:** Outputs are based on AI deductions from historical data and should not be considered investment advice.
*   **Data Accuracy:** Data sources may have inaccuracies or delays; reliability is not guaranteed.
*   **Model Hallucination:** AI models have limitations; we do not guarantee the accuracy, completeness, or timeliness of the information.
*   **Liability:** Developers are not liable for losses from the use of this framework. Investing involves risk.

**Always understand the risks before using this framework for trading decisions.**

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