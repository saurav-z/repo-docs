# ContestTrade: Revolutionizing AI-Driven Trading with a Multi-Agent System

**ContestTrade** is a cutting-edge multi-agent trading framework designed to autonomously identify and capitalize on event-driven investment opportunities, offering a sophisticated approach to AI-powered financial analysis.  [Explore the original repository](https://github.com/FinStep-AI/ContestTrade).

[![arXiv](https://img.shields.io/badge/arXiv-2508.00554-B31B1B?logo=arxiv)](https://arxiv.org/abs/2508.00554)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)](https://www.python.org/downloads/release/python-3100/)
[![Reddit](https://img.shields.io/badge/Reddit-ContestTradeCommunity-orange?logo=reddit&logoColor=white)](https://www.reddit.com/r/ContestTradeCommunity/?feed=home)
[![WeChat](https://img.shields.io/badge/WeChat-ContestTrade-brightgreen?logo=wechat&logoColor=white)](./assets/wechat.png)

[English](README.md) | [中文](README_cn.md)

---

## Key Features

*   **Automated Stock Selection:**  Scans the entire market to generate tradable stock lists, eliminating the need for manual screening.
*   **Event-Driven Strategy:** Focuses on catalysts such as news, announcements, capital flows, and policy changes to identify high-impact opportunities.
*   **Customizable Configuration:**  Allows users to define agent research preferences and strategies, adapting to various investment styles.
*   **Multi-Agent Architecture:** Employs a two-stage pipeline with a dual-contest mechanism to ensure robust and reliable decision-making.

## Introduction

ContestTrade is a powerful multi-agent trading framework centered around event-driven stock selection. The system autonomously discovers, evaluates, and tracks event-driven opportunities to provide actionable asset allocation recommendations without human intervention.

## Architecture

<p align="center">
  <img src="assets/architecture.jpg" style="width: 90%; height: auto;">
</p>

ContestTrade operates in a two-stage process, simulating a financial firm's decision-making. This design utilizes a dual-contest framework to ensure final decisions are driven by the most reliable insights.

1.  **Data Processing Stage:** Raw market data from multiple sources is processed by the **Data Team**. Data Analysis Agents transform the raw data into structured textual factors, which are then evaluated by an internal contest mechanism. The mechanism constructs an optimal "factor portfolio".

2.  **Research and Decision-Making Stage:**  The optimal factor portfolio is then passed to the **Research Team**. Research Agents analyze these factors, submit trading proposals, and undergo a second round of internal contests.  The output is a unified and reliable asset allocation strategy.

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

Alternatively, deploy with [Docker](https://docs.n8n.io/hosting/installation/docker/):

```
docker run -it --rm --name contest_trade -v $(pwd)/config.yaml:/ContestTrade/config.yaml finstep/contesttrade:v1.1
```

## Configuration

Configure API keys and LLM parameters by editing the `config_us.yaml` file.

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

> Note: You must obtain the LLM and VLM API keys. Fill in the URL, API Key, and model name according to your platform.

## Preference Configuration

Customize each Research Agent's "trading belief" by modifying the `contest_trade/config/belief_list.json` file.

*Example 1: Short-Term, Aggressive*
```json
[
  "Focus on short-term event-driven opportunities: prioritize company announcements, M&A and restructuring, sudden order increases, technological breakthroughs and other catalysts; prefer mid/small-cap, high-volatility thematic stocks, suitable for aggressive arbitrage strategies."
]
```

*Example 2: Stable, Conservative*
```json
[
  "Focus on stable, high-certainty events: pay attention to dividends, buybacks, earnings forecast confirmations, major contract signings and policy tailwinds; prefer large-cap blue-chips with low volatility and high certainty, suitable for conservative allocation."
]
```

*Default Configuration*
```json
[
  "Based on the provided information, comprehensively consider each company's business dynamics, industry trends and potential market impact. Recommend stock portfolios with short-term investment potential for two groups: Group 1 — risk-seekers (prefer high volatility, high returns, mid/low market cap stocks); Group 2 — conservative investors (prefer low volatility, stable returns, large-cap stocks).",
  "Based on recent sudden events, policy adjustments and company announcements as catalysts, combined with market sentiment transmission paths and capital game features. Screen event-driven opportunities for two different styles: Group 1 — aggressive arbitrage (prefer restructuring expectation, sudden order increases, technical breakthroughs in small-cap stocks); Group 2 — defensive arbitrage (prefer dividend increases, large buybacks, acquisition of franchise rights in blue-chip stocks). Pay attention to northbound capital movement and institutional seat trends on the trading leaderboard for resonance effects."
]
```

## Usage

Start ContestTrade using the CLI:

```bash
python -m cli.main run
```

<p align="center">
  <img src="assets/contest_trade_cli_select_market.jpg" style="width: 100%; height: auto;">
</p>

View the results summary.

<p align="center">
  <img src="assets/contest_trade_cli_main_us.jpg" style="width: 100%; height: auto;">
</p>

Review detailed research reports.

<p align="center">
  <img src="assets/contest_trade_cli_report_us.jpg" style="width: 100%; height: auto;">
</p>
<p align="center">
  <img src="assets/contest_trade_cli_research_report_us.jpg" style="width: 100%; height: auto;">
</p>

Or, examine in-depth data analysis.

<p align="center">
  <img src="assets/contest_trade_cli_data_report_us.jpg" style="width: 100%; height: auto;">
</p>

> All reports are saved in Markdown format within the `contest_trade/agents_workspace/results` directory.

## 🌟 Vision & Roadmap

The project aims to advance quantitative trading within the AGI era by enhancing infrastructure, adding agents, and creating a trustworthy, scalable agent trading framework.

### Roadmap

**V1.1 (Completed): Framework Enhancements**

*   [✓] Decoupled core data sources for multiple data source adapters (`data-provider` refactor)
*   [✓] Improved CLI logging and user interaction

**V2.0 (Completed): Market & Feature Expansion**

*   [✓] Added access to the **US stock** market.
*   [✓] Introduced new factors and signal sources.

**Future Plans:**

*   [ ] Support for Hong Kong and other stock markets
*   [ ] Visual backtesting and analysis interface
*   [ ] Scalability for increased agent support

## Contributing

Contribute to ContestTrade! See the **[Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md)** for code contributions. Non-code contributions (e.g., feature suggestions, bug reports, and testing feedback) are welcome via the [Issues page](https://github.com/FinStep-AI/ContestTrade/issues).

## Star History

<div align="center">
  <a href="https://star-history.com/#FinStep-AI/ContestTrade&Date">
    <img src="https://api.star-history.com/svg?repos=FinStep-AI/ContestTrade&type=Date" alt="Star History Chart" style="width: 80%;">
  </a>
</div>

## Disclaimer

**Important Disclaimer:** This project is for research and educational purposes only. It does not constitute investment advice.

**Risk Warning:**
*   **Market Risk:** This project does not constitute any form of investment, financial, legal, or tax advice. All outputs, including trading signals and analyses, are the results of AI model deductions based on historical data and should not be considered a basis for any buy or sell operations.
*   **Data Accuracy:** Data sources may have delays, inaccuracies, or incompleteness. Reliability is not guaranteed.
*   **Model Hallucination:** AI models have limitations and are subject to the risk of "hallucination." Accuracy, completeness, and timeliness of information are not guaranteed.
*   **Liability:** The developers are not liable for any losses resulting from the use of or inability to use this framework. Investing involves risks; exercise caution.

**Understand the risks before making trading decisions.**

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

Licensed under the [Apache 2.0 License](LICENSE).