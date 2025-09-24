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

**ContestTrade** is a cutting-edge, open-source multi-agent trading framework designed to build autonomous AI trading teams for event-driven investment strategies.

## Key Features

*   **Automated Stock Selection:**  Scans the entire market to automatically identify and generate tradable stock lists, eliminating manual screening.
*   **Event-Driven Strategy:** Focuses on opportunities triggered by catalysts like news, announcements, capital flows, and policy changes, capitalizing on significant information impact.
*   **Personalized Configuration:** Offers flexible customization with user-defined agent research preferences and strategies to suit diverse investment styles.
*   **Multi-Agent Architecture:** Employs a two-stage contest mechanism to refine data and strategies, ensuring robust and effective trading signals.
*   **Multi-Market Support:** Supports the US stock market and plans for expansion to include other markets (e.g., Hong Kong).

## Introduction

ContestTrade is an innovative multi-agent (Multi-Agent) trading framework, specifically designed for event-driven stock selection. The system automatically discovers, evaluates, and tracks event-driven opportunities with investment potential, delivering executable asset allocation recommendations without human intervention.

## Framework Overview

<p align="center">
  <img src="assets/architecture.jpg" style="width: 90%; height: auto;">
</p>

The ContestTrade workflow operates through a two-stage pipeline, mimicking the dynamic decision-making process of an investment firm. This dual-contest framework ensures decisions are based on the most reliable insights, maintaining adaptability and resilience in complex markets:

1.  **Data Processing Stage:**  Raw market data from multiple sources is processed by the **Data Team**, where multiple Data Analysis Agents convert data into structured "textual factors".  An internal contest mechanism evaluates these factors to construct an optimal "factor portfolio."
2.  **Research and Decision-Making Stage:** The factor portfolio is then passed to the **Research Team**. Research Agents, each with unique "Trading Beliefs" and access to financial tools, analyze the factors and submit trading proposals. A second contest evaluates these proposals, synthesizing a unified asset allocation strategy.

## Installation

Get started with ContestTrade by following these steps:

```bash
# 1. Clone the repository
git clone https://github.com/FinStep-AI/ContestTrade.git
cd ContestTrade

# 2. Create and activate a virtual environment (recommended)
conda create -n contesttrade python=3.10
conda activate contesttrade

# 3. Install dependencies
pip install -r requirements.txt
```

Alternatively, deploy with Docker:

```bash
docker run -it --rm --name contest_trade -v $(pwd)/config.yaml:/ContestTrade/config.yaml finstep/contesttrade:v2.0
```

## Configuration

Configure ContestTrade by editing the `config_us.yaml` file with your API keys.

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

> **Note:** Obtain your own LLM and VLM API keys, entering the appropriate URL, API key, and model name.

## Preference Customization

Customize Research Agents by modifying their "trading beliefs" in `contest_trade/config/belief_list.json`.  Each agent's belief generates up to five signals.

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

The CLI provides an interactive interface for market selection and result review.
<p align="center">
  <img src="assets/contest_trade_cli_select_market.jpg" style="width: 100%; height: auto;">
</p>

View agent signals in the results summary after completion.
<p align="center">
  <img src="assets/contest_trade_cli_main_us.jpg" style="width: 100%; height: auto;">
</p>

Access detailed research and data analysis reports.
<p align="center">
  <img src="assets/contest_trade_cli_report_us.jpg" style="width: 100%; height: auto;">
</p>
<p align="center">
  <img src="assets/contest_trade_cli_research_report_us.jpg" style="width: 100%; height: auto;">
</p>
<p align="center">
  <img src="assets/contest_trade_cli_data_report_us.jpg" style="width: 100%; height: auto;">
</p>

> All reports are saved in the `contest_trade/agents_workspace/results` directory in Markdown format for easy reference and sharing.

## 🌟 Vision & Roadmap

ContestTrade aims to leverage the power of the open-source community to explore new paradigms of quantitative trading in the era of AGI.

### Roadmap

**V1.1 (Finished):** Framework Stability Enhancement & Core Experience Optimization

*   [✓] Decoupled core data source module for multi-source data adaptation (`data-provider` refactor)
*   [✓] Enhanced CLI logging and interaction

**V2.0 (Finished):** Market and Function Expansion

*   [✓] Access to US stock market data.
*   [✓] Expanded factors and signal sources.

**Future Plans:**

*   [ ] Support for Hong Kong and other stock markets.
*   [ ] Visual backtesting and analysis interface.
*   [ ] Scalable agent support.

## Contributing

ContestTrade thrives on community contributions!  All contributions are welcome.

*   **Developers:** See the **[Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md)**.
*   **Non-Code Contributions:**  Propose features, report bugs via the [Issues page](https://github.com/FinStep-AI/ContestTrade/issues), and provide feedback.

## Star History

<div align="center">
  <a href="https://star-history.com/#FinStep-AI/ContestTrade&Date">
    <img src="https://api.star-history.com/svg?repos=FinStep-AI/ContestTrade&type=Date" alt="Star History Chart" style="width: 80%;">
  </a>
</div>

## Disclaimer

**Important:**  ContestTrade is an open-source research framework for academic and educational purposes only.  It does not provide investment advice.

**Risk Warning:**

*   **Market Risk:** The project does not offer investment, financial, legal, or tax advice. All outputs are based on AI model deductions from historical data and should not be used for buy/sell decisions.
*   **Data Accuracy:** Data may have delays, inaccuracies, or be incomplete. Data reliability is not guaranteed.
*   **Model Limitations:** AI models can experience "hallucination." We do not guarantee the accuracy, completeness, or timeliness of information generated.
*   **Liability:** Developers are not liable for any losses from using or not being able to use the framework. Investing carries risks; proceed with caution.

**Understand the risks before using this framework for trading.**

## Citation

Cite our paper if you use ContestTrade in your research:

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

**[Back to the GitHub Repository](https://github.com/FinStep-AI/ContestTrade)**