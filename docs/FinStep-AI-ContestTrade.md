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

# ContestTrade: AI-Powered Multi-Agent Trading for Event-Driven Stock Selection

**ContestTrade** revolutionizes quantitative trading by employing a multi-agent framework to automatically identify and capitalize on event-driven investment opportunities.  [Explore the original repository on GitHub](https://github.com/FinStep-AI/ContestTrade).

## Key Features

*   **Automated Stock Selection:** Automatically scans the entire market and generates tradable stock lists, eliminating manual screening.
*   **Event-Driven Focus:**  Identifies opportunities triggered by market catalysts such as news, announcements, and policy changes.
*   **Personalized Strategy:** Supports user-defined agent preferences and investment strategies to adapt to diverse investment styles.
*   **Multi-Agent Architecture:** Employs a two-stage pipeline to simulate investment firm decision-making, enhancing adaptability.
*   **US Stock Market Support:** Provides access to US stock market data.

## Introduction

ContestTrade is a cutting-edge multi-agent trading framework designed for event-driven stock selection.  The system autonomously discovers, evaluates, and tracks event-driven opportunities with investment potential, culminating in actionable asset allocation recommendations.

## Framework Overview

<p align="center">
  <img src="assets/architecture.jpg" style="width: 90%; height: auto;">
</p>

ContestTrade operates through a two-stage pipeline:

1.  **Data Processing Stage:**  Raw market data is processed by the **Data Team**. Data Analysis Agents refine this data into structured "textual factors." An internal contest mechanism evaluates these factors to create an optimal "factor portfolio."
2.  **Research and Decision-Making Stage:** The factor portfolio is passed to the **Research Team**. Research Agents, each with unique "Trading Beliefs" and access to financial tools, analyze factors and submit trading proposals. A second contest synthesizes these proposals into a final asset allocation strategy.

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

Configure API keys and LLM parameters in `config_us.yaml`.

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

> Note: Obtain LLM and VLM API keys. Provide the URL, API Key, and model name based on your chosen platform and model.

## Preference Customization

Customize Research Agent "trading beliefs" in `contest_trade/config/belief_list.json` (JSON array of strings).

Example 1 (Aggressive):
```json
[
  "Focus on short-term event-driven opportunities: prioritize company announcements, M&A and restructuring, sudden order increases, technological breakthroughs and other catalysts; prefer mid/small-cap, high-volatility thematic stocks, suitable for aggressive arbitrage strategies."
]
```

Example 2 (Conservative):
```json
[
  "Focus on stable, high-certainty events: pay attention to dividends, buybacks, earnings forecast confirmations, major contract signings and policy tailwinds; prefer large-cap blue-chips with low volatility and high certainty, suitable for conservative allocation."
]
```

Default Configuration:
```json
[
  "Based on the provided information, comprehensively consider each company's business dynamics, industry trends and potential market impact. Recommend stock portfolios with short-term investment potential for two groups: Group 1 — risk-seekers (prefer high volatility, high returns, mid/low market cap stocks); Group 2 — conservative investors (prefer low volatility, stable returns, large-cap stocks).",
  "Based on recent sudden events, policy adjustments and company announcements as catalysts, combined with market sentiment transmission paths and capital game features. Screen event-driven opportunities for two different styles: Group 1 — aggressive arbitrage (prefer restructuring expectation, sudden order increases, technical breakthroughs in small-cap stocks); Group 2 — defensive arbitrage (prefer dividend increases, large buybacks, acquisition of franchise rights in blue-chip stocks). Pay attention to northbound capital movement and institutional seat trends on the trading leaderboard for resonance effects."
]
```

## Usage

Run ContestTrade using the CLI:

```bash
python -m cli.main run
```

Follow the interactive terminal to select the market.

<p align="center">
  <img src="assets/contest_trade_cli_select_market.jpg" style="width: 100%; height: auto;">
</p>

View agent signals in the results summary:

<p align="center">
  <img src="assets/contest_trade_cli_main_us.jpg" style="width: 100%; height: auto;">
</p>

View detailed research reports:

<p align="center">
  <img src="assets/contest_trade_cli_report_us.jpg" style="width: 100%; height: auto;">
</p>

<p align="center">
  <img src="assets/contest_trade_cli_research_report_us.jpg" style="width: 100%; height: auto;">
</p>

View detailed data analysis reports:

<p align="center">
  <img src="assets/contest_trade_cli_data_report_us.jpg" style="width: 100%; height: auto;">
</p>

Reports are saved in `contest_trade/agents_workspace/results` in Markdown format.

## 🌟 Vision & Roadmap

The project aims to explore new quantitative trading paradigms in the AGI era.

### Roadmap

**V1.1 (Finished): Framework Stability & Core Experience**
- [✓] Decoupled core data source module (`data-provider` refactor).
- [✓] Improved CLI logging and interaction.

**V2.0 (Finished): Market and Function Expansion**
- [✓] Access to US stock market data.
- [✓] Expanded factors and signal sources.

**Future Plans:**
- [ ] Support for Hong Kong and other markets
- [ ] Visual backtesting and analysis interface
- [ ] Scalable agent support

## Contributing

Contributions are welcome!  See the **[Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md)**.

We also value non-code contributions:
*   [Issues Page](https://github.com/FinStep-AI/ContestTrade/issues) for feature suggestions and bug reports.
*   Testing results and user experience feedback.

## Star History

<div align="center">
  <a href="https://star-history.com/#FinStep-AI/ContestTrade&Date">
    <img src="https://api.star-history.com/svg?repos=FinStep-AI/ContestTrade&type=Date" alt="Star History Chart" style="width: 80%;">
  </a>
</div>

## Disclaimer

**Important Disclaimer:** This project is for academic and educational purposes and does not constitute investment advice.

**Risk Warning:**
*   **Market Risk:** This project does not offer investment, financial, legal, or tax advice.
*   **Data Accuracy:** Data sources may have delays or inaccuracies.
*   **Model Hallucination:** AI models have limitations and may "hallucinate."
*   **Liability:** Developers assume no liability for losses from using this framework.

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