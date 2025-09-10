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
# ContestTrade: Revolutionizing AI-Driven Trading

**ContestTrade is a cutting-edge, multi-agent trading framework designed to autonomously identify and capitalize on event-driven investment opportunities.**  For the original source code, please visit the [ContestTrade GitHub Repository](https://github.com/FinStep-AI/ContestTrade).

## Key Features

*   **Automated Stock Selection:** Scans the entire market to generate tradable stock lists, eliminating manual screening.
*   **Event-Driven Strategy:** Focuses on opportunities triggered by catalysts like news, announcements, and policy changes.
*   **Personalized Configuration:** Enables user-defined agent research preferences and strategies, adapting to various investment styles.
*   **Multi-Agent Architecture:**  Employs a two-stage pipeline with data and research teams to simulate investment firm decision-making.
*   **Open-Source & Community Driven:** Encourages contributions for infrastructure, agent variety, and financial trading exploration.

## Introduction

ContestTrade is a sophisticated, multi-agent trading framework engineered for event-driven stock selection. It's designed to automatically discover, evaluate, and track investment opportunities without human intervention, ultimately providing actionable asset allocation recommendations.

## Framework Overview

<p align="center">
  <img src="assets/architecture.jpg" style="width: 90%; height: auto;">
</p>

The ContestTrade framework operates through a two-stage pipeline, mirroring the investment decision-making process of a financial firm.

1.  **Data Processing Stage:** Raw market data is processed by the **Data Team**, consisting of Data Analysis Agents that convert data into "textual factors." An internal contest mechanism evaluates these factors to construct an optimal "factor portfolio."

2.  **Research and Decision-Making Stage:** The factor portfolio is passed to the **Research Team**, comprising Research Agents with unique "Trading Beliefs." These agents conduct in-depth analysis and submit trading proposals. A second internal contest then synthesizes these proposals into a unified asset allocation strategy.

## Installation

Follow these steps to install and set up ContestTrade:

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

Or, deploy with [Docker](https://docs.n8n.io/hosting/installation/docker/):

```
docker run -it --rm --name contest_trade -v $(pwd)/config.yaml:/ContestTrade/config.yaml finstep/contesttrade:v1.1
```

## Configuration

Before running ContestTrade, configure your API keys and LLM parameters.  Edit the `config_us.yaml` file with the necessary keys.

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

> **Note:**  You will need to obtain your own LLM and VLM API keys and populate the corresponding URLs, API keys, and model names.

## Preference Configuration

Each Research Agent operates according to a defined "trading belief."  These beliefs guide the system's generation of investment signals, based on data and tools (each belief yields up to 5 signals). The configuration file, located at `contest_trade/config/belief_list.json`, uses a JSON array of strings.

**Example Preferences:**

*   **Short-term event-driven (Aggressive):**
    ```json
    [
      "Focus on short-term event-driven opportunities: prioritize company announcements, M&A and restructuring, sudden order increases, technological breakthroughs and other catalysts; prefer mid/small-cap, high-volatility thematic stocks, suitable for aggressive arbitrage strategies."
    ]
    ```

*   **Stable events (Conservative):**
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

Launch ContestTrade via the Command Line Interface (CLI).

```bash
python -m cli.main run
```

Follow the interactive terminal prompts to select the market for analysis.  The default analysis time is the current time.

<p align="center">
  <img src="assets/contest_trade_cli_select_market.jpg" style="width: 100%; height: auto;">
</p>

View the signals provided by the agents in the results summary after the analysis.

<p align="center">
  <img src="assets/contest_trade_cli_main_us.jpg" style="width: 100%; height: auto;">
</p>

Explore detailed research reports.

<p align="center">
  <img src="assets/contest_trade_cli_report_us.jpg" style="width: 100%; height: auto;">
</p>

<p align="center">
  <img src="assets/contest_trade_cli_research_report_us.jpg" style="width: 100%; height: auto;">
</p>

Access detailed data analysis reports.

<p align="center">
  <img src="assets/contest_trade_cli_data_report_us.jpg" style="width: 100%; height: auto;">
</p>

> All reports are saved in Markdown format within the `contest_trade/agents_workspace/results` directory for your review and sharing.

## 🌟 Vision & Roadmap

ContestTrade aims to leverage open-source collaboration to explore new paradigms in quantitative trading within the AGI era.

### Roadmap

**V1.1 (Completed):** Framework stability and core experience improvements.
-   [✓] Decoupled data source module.
-   [✓] Optimized CLI logging and interaction.

**V2.0 (Completed):** Market and Feature Expansion
-   [✓] Support for US stock market data.
-   [✓] Expanded factors and signal sources.

**Future Plans:**
-   [ ] Support for Hong Kong and other markets.
-   [ ] Visual backtesting and analysis interface.
-   [ ] Support for scaling up more agents.

## Contributing

ContestTrade thrives on community contributions.  All contributions are welcome!

*   **Developers:** Consult the **[Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md)** to learn how to contribute code.

*   **Non-Code Contributions:**
    *   Suggest feature ideas or report bugs: [Issues Page](https://github.com/FinStep-AI/ContestTrade/issues)
    *   Provide feedback on test results and user experience.

## Star History

<div align="center">
  <a href="https://star-history.com/#FinStep-AI/ContestTrade&Date">
    <img src="https://api.star-history.com/svg?repos=FinStep-AI/ContestTrade&type=Date" alt="Star History Chart" style="width: 80%;">
  </a>
</div>

## Disclaimer

**Important:** This project, `ContestTrade`, is for academic and educational purposes only. The project's output is not financial advice.

**Risk Warning:**

*   **Market Risk:**  The project's output is *not* investment advice.  AI model outputs are based on historical data and should not drive any buy/sell decisions.
*   **Data Accuracy:** Data sources may have inaccuracies and delays. Data reliability is not guaranteed.
*   **Model Hallucination:** AI models have limitations, including "hallucination" risk.  Accuracy, completeness, and timeliness of generated information are not guaranteed.
*   **Liability:**  The developers are not liable for losses resulting from the use of this framework. Investing carries risks.

**Prior to any trading decisions, be sure to fully understand the associated risks.**

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

This project is licensed under the [Apache 2.0 License](LICENSE).