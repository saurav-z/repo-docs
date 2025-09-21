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

# ContestTrade: Automated AI-Powered Multi-Agent Trading Framework

**ContestTrade** revolutionizes AI-driven trading by employing a multi-agent system to automatically analyze markets and identify event-driven investment opportunities. Explore the project on [GitHub](https://github.com/FinStep-AI/ContestTrade)!

## Key Features

*   **Autonomous Stock Selection:** Automates whole-market scanning to generate tradable stock lists without manual intervention.
*   **Event-Driven Strategy:** Focuses on catalysts like news, announcements, and policy changes to identify impactful opportunities.
*   **Customizable Configuration:** Allows user-defined agent preferences and strategies for adaptable investment styles.

## Framework Overview

ContestTrade utilizes a two-stage pipeline mimicking an investment firm's decision-making process.  This design ensures robust and effective insights for trading.

<p align="center">
  <img src="assets/architecture.jpg" style="width: 90%; height: auto;">
</p>

1.  **Data Processing Stage:**  The **Data Team** processes market data, with Data Analysis Agents converting raw data into structured "textual factors." An internal contest mechanism then evaluates these factors to create an optimal "factor portfolio."

2.  **Research and Decision-Making Stage:** The **Research Team** receives the factor portfolio, and multiple Research Agents, each with unique "Trading Beliefs" and access to financial tools, analyze the factors and submit trading proposals. A second internal contest synthesizes these proposals into a final, reliable asset allocation strategy.

## Installation

Get started with ContestTrade in a few simple steps:

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

Configure ContestTrade by adding your API keys and parameters in the `config_us.yaml` file.

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

> Note: You need to apply for the LLM API and VLM API by yourself. Fill in the URL, API Key, and model name according to the platform and model you use.

## Preference: Define Your Trading Beliefs

Customize your agent's trading behavior by modifying the `contest_trade/config/belief_list.json` file. This file contains a JSON array of strings, each defining a "trading belief".

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

Run ContestTrade from the command line:

```bash
python -m cli.main run
```

Follow the terminal prompts to select your market and view the results:

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

>  Reports are saved in the `contest_trade/agents_workspace/results` directory in Markdown format.

## Vision & Roadmap

The project aims to advance quantitative trading in the AGI era by building robust infrastructure, diverse agents, and a scalable trading framework.

### Roadmap

*   **V1.1 (Completed):** Framework Stability and Core Experience Improvements.
    *   \[✓] Decoupled data source module for multiple data source adaptors.
    *   \[✓] Improved CLI logging and interaction.
*   **V2.0 (Completed):** Market and Function Expansion.
    *   \[✓]  US Stock Market Data Support.
    *   \[✓] Enhanced factors and signal sources.
*   **Future Plans:**
    *   \[ ] Support for Hong Kong and other stock markets
    *   \[ ] Visual backtesting and analysis interface
    *   \[ ] Support for scaling up more agents

## Contributing

Contribute to ContestTrade!  We welcome all contributions.

*   **Developer Contributions:** Refer to the **[Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md)**.
*   **Non-Code Contributions:** Propose feature suggestions, report bugs via the [Issues page](https://github.com/FinStep-AI/ContestTrade/issues), or provide testing feedback.

## Star History

<div align="center">
  <a href="https://star-history.com/#FinStep-AI/ContestTrade&Date">
    <img src="https://api.star-history.com/svg?repos=FinStep-AI/ContestTrade&type=Date" alt="Star History Chart" style="width: 80%;">
  </a>
</div>

## Disclaimer

**Important Disclaimer:** This project is an open-source research framework for academic and educational purposes only.  It does not constitute investment advice.

**Risk Warning:**
*   **Market Risk:** The framework's outputs, including trading signals, are based on AI models and historical data, and should not be used as the basis for investment decisions.
*   **Data Accuracy:**  Data sources may be subject to delays, inaccuracies, and incompleteness.
*   **Model Limitations:** AI models, including LLMs, have limitations, including the risk of "hallucination."
*   **Liability:**  The developers are not liable for any losses from the use or inability to use this framework. Investing involves risks.

**Always understand the risks before using this framework for trading decisions.**

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