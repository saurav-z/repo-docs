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
# ContestTrade: Your AI-Powered Trading Assistant for Event-Driven Investing

**ContestTrade** is an innovative, multi-agent trading framework designed to identify and capitalize on event-driven investment opportunities, automating the entire trading process.  Explore the power of AI in financial markets.  [Visit the original repository](https://github.com/FinStep-AI/ContestTrade) to learn more.

## Key Features

*   **Automated Stock Selection:**  Scans the entire market and automatically generates a list of tradable stocks, eliminating manual screening.
*   **Event-Driven Strategy:** Focuses on catalysts like news, announcements, and policy changes to find high-impact opportunities.
*   **Customizable Agent Preferences:** Tailor research preferences and trading strategies to match your specific investment style.
*   **Multi-Agent Architecture:** Utilizes a two-stage pipeline with data and research teams to mimic decision-making in an investment firm.

## Introduction

ContestTrade is a pioneering multi-agent trading framework, expertly designed for event-driven stock selection. This system automatically discovers, evaluates, and tracks event-driven opportunities with investment potential, providing executable asset allocation recommendations without human intervention.

## Framework Overview

<p align="center">
  <img src="assets/architecture.jpg" style="width: 90%; height: auto;">
</p>

ContestTrade's workflow is structured around a two-stage pipeline, designed to mimic the dynamic decision-making process of an investment firm. This dual-contest framework ensures that final decisions are driven by the most effective insights, enhancing adaptability and resilience in complex markets.

1.  **Data Processing Stage:** The **Data Team** ingests raw market data from multiple sources. Multiple Data Analysis Agents convert this raw data into structured "textual factors," which are then evaluated via an internal contest mechanism to create an "optimal factor portfolio."

2.  **Research and Decision-Making Stage:** The optimal factor portfolio is passed to the **Research Team**. Research Agents, each with unique "Trading Beliefs" and financial tools, analyze the factors and submit trading proposals. Another internal contest evaluates these proposals, resulting in a unified and reliable asset allocation strategy.

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
docker run -it --rm --name contest_trade -v $(pwd)/config.yaml:/ContestTrade/config.yaml finstep/contesttrade:v2.0
```

## Configuration

Configure ContestTrade by providing the necessary API keys and LLM parameters. Edit the `config_us.yaml` file, entering your API keys.

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

> Note: Apply for the LLM API and VLM API and populate the URL, API Key, and model name accordingly.

## Preference

Each Research Agent has a "trading belief". The system generates investment signals based on these beliefs, combined with data and tools. Configure these beliefs in `contest_trade/config/belief_list.json`.

Example 1 — preference for short-term event-driven (more aggressive):
```json
[
  "Focus on short-term event-driven opportunities: prioritize company announcements, M&A and restructuring, sudden order increases, technological breakthroughs and other catalysts; prefer mid/small-cap, high-volatility thematic stocks, suitable for aggressive arbitrage strategies."
]
```

Example 2 — preference for stable events (more conservative):
```json
[
  "Focus on stable, high-certainty events: pay attention to dividends, buybacks, earnings forecast confirmations, major contract signings and policy tailwinds; prefer large-cap blue-chips with low volatility and high certainty, suitable for conservative allocation."
]
```

Default configuration:
```json
[
  "Based on the provided information, comprehensively consider each company's business dynamics, industry trends and potential market impact. Recommend stock portfolios with short-term investment potential for two groups: Group 1 — risk-seekers (prefer high volatility, high returns, mid/low market cap stocks); Group 2 — conservative investors (prefer low volatility, stable returns, large-cap stocks).",
  "Based on recent sudden events, policy adjustments and company announcements as catalysts, combined with market sentiment transmission paths and capital game features. Screen event-driven opportunities for two different styles: Group 1 — aggressive arbitrage (prefer restructuring expectation, sudden order increases, technical breakthroughs in small-cap stocks); Group 2 — defensive arbitrage (prefer dividend increases, large buybacks, acquisition of franchise rights in blue-chip stocks). Pay attention to northbound capital movement and institutional seat trends on the trading leaderboard for resonance effects."
]
```

## Usage

Start ContestTrade via the CLI:

```bash
python -m cli.main run
```

This opens an interactive terminal interface where you select the market to analyze.

<p align="center">
  <img src="assets/contest_trade_cli_select_market.jpg" style="width: 100%; height: auto;">
</p>

View the signals after the agents complete their run.

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

> Reports are saved in the `contest_trade/agents_workspace/results` directory in Markdown format.

## 🌟 Vision & Roadmap

ContestTrade aims to leverage open-source collaboration to pioneer new quantitative trading approaches in the AGI era, striving for a stable, trustworthy, and scalable framework.

### Roadmap

**V1.1 (Finished): Framework Stability Enhancement & Core Experience Optimization**
- [✓] Decoupled core data source module with adaptors for multiple data sources. (`data-provider` refactor)
- [✓] Optimized CLI logging and interaction experience.

**V2.0 (Finished): Market and Function Expansion**
- [✓] Access to **US stock** market data.
- [✓] Introduced richer factors and signal sources.

**Future Plans:**
- [ ] Support for Hong Kong stocks and other markets
- [ ] Visual backtesting and analysis interface
- [ ] Support for scaling up more agents

## Contributing

Join our community! We welcome contributions of all kinds. See the **[Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md)** for details on how to contribute.

We value both code and non-code contributions, including:
*   Proposing features or reporting bugs: [Go to Issues page](https://github.com/FinStep-AI/ContestTrade/issues)
*   Providing feedback on your testing results and user experience.

## Star History

<div align="center">
  <a href="https://star-history.com/#FinStep-AI/ContestTrade&Date">
    <img src="https://api.star-history.com/svg?repos=FinStep-AI/ContestTrade&type=Date" alt="Star History Chart" style="width: 80%;">
  </a>
</div>

## Disclaimer

**Important Disclaimer:** ContestTrade is for academic and educational use only and does not constitute investment advice.

**Risk Warning:**
*   **Market Risk:** This project is for informational and educational purposes and should not be taken as investment, financial, legal, or tax advice. Trading signals and analyses are based on AI model deductions and should not be the basis for investment decisions.
*   **Data Accuracy:** Data used by the framework may be subject to delays, inaccuracies, or incompleteness.
*   **Model Hallucination:** AI models have limitations and can produce inaccurate or unreliable information.
*   **Liability:** The developers are not liable for any losses resulting from the use or inability to use this framework.
**Understand the risks before making any actual trading decisions.**

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