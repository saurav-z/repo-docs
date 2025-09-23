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

# ContestTrade: Automate Your Event-Driven Trading with AI

**ContestTrade** is an innovative, open-source, multi-agent trading framework designed to automate event-driven stock selection, offering a powerful toolkit for AI-driven investment strategies; find out more on the [original repo](https://github.com/FinStep-AI/ContestTrade).

## Key Features

*   **Automated Stock Selection:** Automatically scans the entire market and generates tradable stock lists, eliminating manual screening.
*   **Event-Driven Strategy:** Focuses on opportunities triggered by news, announcements, capital flows, and policy changes.
*   **Personalized Configuration:**  Supports user-defined agent research preferences and strategies to adapt to diverse investment styles.
*   **Multi-Agent Framework:** Employs a two-stage pipeline with a dual-contest mechanism to ensure robust and reliable investment recommendations.

## Introduction

ContestTrade is a sophisticated multi-agent trading framework specializing in event-driven stock selection. It is engineered to automatically discover, evaluate, and track event-driven opportunities with investment value without human intervention. The ultimate goal is to generate executable asset allocation recommendations.

## Framework Overview

<p align="center">
  <img src="assets/architecture.jpg" style="width: 90%; height: auto;">
</p>

ContestTrade's workflow mimics a professional investment firm's decision-making process, utilizing a structured two-stage pipeline:

1.  **Data Processing Stage:** The **Data Team** processes raw market data from multiple sources. Data Analysis Agents transform this into structured "textual factors." An internal contest mechanism then evaluates these factors, constructing an optimal "factor portfolio."

2.  **Research and Decision-Making Stage:**  The optimal factor portfolio is then passed to the **Research Team**. Multiple Research Agents, each with their unique "Trading Beliefs" and financial tools, conduct in-depth analyses and submit trading proposals. A second internal contest synthesizes these proposals into a unified and reliable asset allocation strategy.

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

Alternatively, deploy with [Docker](https://docs.n8n.io/hosting/installation/docker/):

```
docker run -it --rm --name contest_trade -v $(pwd)/config.yaml:/ContestTrade/config.yaml finstep/contesttrade:v2.0
```

## Configuration

Before running ContestTrade, configure your API keys and LLM parameters in `config_us.yaml`:

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

> Note: Obtain your LLM and VLM API keys and enter the correct URL, API Key, and model name.

## Preference Configuration

Each Research Agent's "trading belief" influences the investment signals generated, configured in `contest_trade/config/belief_list.json`:

Example 1 — Short-Term, Aggressive:
```json
[
  "Focus on short-term event-driven opportunities: prioritize company announcements, M&A and restructuring, sudden order increases, technological breakthroughs and other catalysts; prefer mid/small-cap, high-volatility thematic stocks, suitable for aggressive arbitrage strategies."
]
```

Example 2 — Stable, Conservative:
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

Use the Command Line Interface (CLI) to run ContestTrade:

```bash
python -m cli.main run
```

The CLI allows market selection, with the default analysis time set to the current time.

<p align="center">
  <img src="assets/contest_trade_cli_select_market.jpg" style="width: 100%; height: auto;">
</p>

View the results summary after agent execution:

<p align="center">
  <img src="assets/contest_trade_cli_main_us.jpg" style="width: 100%; height: auto;">
</p>

Access detailed research and data analysis reports:

<p align="center">
  <img src="assets/contest_trade_cli_report_us.jpg" style="width: 100%; height: auto;">
</p>
<p align="center">
  <img src="assets/contest_trade_cli_research_report_us.jpg" style="width: 100%; height: auto;">
</p>
<p align="center">
  <img src="assets/contest_trade_cli_data_report_us.jpg" style="width: 100%; height: auto;">
</p>

> All reports are saved in the `contest_trade/agents_workspace/results` directory in Markdown format.

## Vision & Roadmap

We are committed to exploring the potential of AI in quantitative trading and creating a robust, scalable agent trading framework.

### Roadmap

**V1.1 (Completed):** Framework Stability and Experience Optimization
- [✓]  Decoupled core data source module. (`data-provider` refactor)
- [✓]  Optimized CLI logging and interaction.

**V2.0 (Completed):** Market and Feature Expansion
- [✓]  Added US stock market data access.
- [✓]  Introduced more factors and signal sources.

**Future Plans:**
- [ ] Support for Hong Kong stocks and other markets
- [ ] Visual backtesting and analysis interface
- [ ] Scaling up agent capabilities

## Contributing

We welcome contributions!  Refer to the **[Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md)** for development guidelines.

Non-code contributions are also valued:
*   **Feature suggestions and bug reports:** [Go to Issues page](https://github.com/FinStep-AI/ContestTrade/issues)
*   **Feedback:** Share your testing results and user experiences.

## Star History

<div align="center">
  <a href="https://star-history.com/#FinStep-AI/ContestTrade&Date">
    <img src="https://api.star-history.com/svg?repos=FinStep-AI/ContestTrade&type=Date" alt="Star History Chart" style="width: 80%;">
  </a>
</div>

## Disclaimer

**Important Disclaimer:** This project, `ContestTrade`, is for academic and educational purposes only.  It is *not* investment advice.

**Risk Warning:**
*   **Market Risk:** Outputs are AI-driven and based on historical data. They are *not* financial, legal, or tax advice.
*   **Data Accuracy:** Data may have delays, inaccuracies, or be incomplete.
*   **Model Hallucination:** AI models have limitations, including "hallucination."  We do not guarantee accuracy.
*   **Liability:** Developers are not liable for any losses from using this framework. Investing involves risk.

**Fully understand the risks before using this framework for any trading decisions.**

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