# ContestTrade: Revolutionizing AI-Driven Trading with Multi-Agent Systems

**ContestTrade** is a cutting-edge, open-source multi-agent trading framework that leverages an innovative internal contest mechanism to build and optimize AI-powered trading strategies. [Check out the original repo](https://github.com/FinStep-AI/ContestTrade)

[![arXiv](https://img.shields.io/badge/arXiv-2508.00554-B31B1B?logo=arxiv)](https://arxiv.org/abs/2508.00554)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)](https://www.python.org/downloads/release/python-3100/)
[![Reddit](https://img.shields.io/badge/Reddit-ContestTradeCommunity-orange?logo=reddit&logoColor=white)](https://www.reddit.com/r/ContestTradeCommunity/?feed=home)
[![WeChat](https://img.shields.io/badge/WeChat-ContestTrade-brightgreen?logo=wechat&logoColor=white)](./assets/wechat.png)

[English](README.md) | [中文](README_cn.md)

---

## Key Features

*   **Automated Stock Selection:**  Scan the entire market and automatically generate a list of tradable stocks, removing manual screening.
*   **Event-Driven Strategy:** Focuses on market catalysts, such as news, announcements, and policy changes, for high-impact investment opportunities.
*   **Customizable Configuration:**  Allows users to personalize agent research preferences and strategies for diverse investment styles.
*   **Multi-Agent Architecture:** Employs a two-stage pipeline with a data processing and research/decision-making phase to simulate an investment firm's decision-making process.

## Introduction

ContestTrade is a multi-agent trading framework focused on event-driven stock selection. The system's goal is to automatically discover, evaluate, and track event-driven opportunities with investment value without human intervention, and finally output executable asset allocation recommendations.

## Framework Overview

<p align="center">
  <img src="assets/architecture.jpg" style="width: 90%; height: auto;">
</p>

The framework utilizes a two-stage process, designed to replicate an investment firm's dynamic decision-making. This dual-contest system ensures that the most robust insights guide the final decisions, providing strong adaptability and resilience in complex markets.

1.  **Data Processing Stage:** The **Data Team** takes in market data from many sources. The data is refined into "textual factors" using several Data Analysis Agents working in parallel. The internal contest mechanism assesses the worth of each data agent's generated factors to create an optimal "factor portfolio."

2.  **Research and Decision-Making Stage:** The **Research Team** receives this factor portfolio. Each of the Research Agents, which have their own unique "Trading Beliefs" and access to powerful financial tools, analyzes these factors in depth. The trading proposals from each agent are then assessed in a second round of internal contests, resulting in a unified asset allocation strategy as the final output.

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

Before starting ContestTrade, set up your API keys and LLM parameters.

Edit the `config_us.yaml` file and input your API keys. Required and optional keys are listed below:

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

## Preference

Each Research Agent uses a "trading belief." The system generates investment signals based on these beliefs combined with data and tools (each belief outputs up to 5 signals).  The configuration file is located at `contest_trade/config/belief_list.json`, and its format is a JSON array of strings.

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

Use the Command Line Interface (CLI) to run ContestTrade.

```bash
python -m cli.main run
```

The terminal interface launches after starting the program. Here, you can pick which market to analyze. The default analysis time is the present time.
<p align="center">
  <img src="assets/contest_trade_cli_select_market.jpg" style="width: 100%; height: auto;">
</p>

You can see the signals provided by the agents in the results summary after the agents finish running.
<p align="center">
  <img src="assets/contest_trade_cli_main_us.jpg" style="width: 100%; height: auto;">
</p>

You can choose to view a more detailed research report.
<p align="center">
  <img src="assets/contest_trade_cli_report_us.jpg" style="width: 100%; height: auto;">
</p>

<p align="center">
  <img src="assets/contest_trade_cli_research_report_us.jpg" style="width: 100%; height: auto;">
</p>

You can also choose to view a more detailed data analysis report.
<p align="center">
  <img src="assets/contest_trade_cli_data_report_us.jpg" style="width: 100%; height: auto;">
</p>

> All the above reports will be saved in the `contest_trade/agents_workspace/results` directory in Markdown format for your future reference and sharing.

## 🌟 Vision & Roadmap

We believe that the era of AGI is quickly approaching. We hope to use the open-source community's power to explore new frontiers in quantitative trading during the AGI era.

The project is focused on developing a more robust infrastructure, expanding the number of agents, exploring AI's limits in financial trading, and building a reliable, scalable agent trading framework.

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

ContestTrade is a community-driven open-source project. Any contribution is welcome!

Read the **[Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md)** to become a ContestTrade contributor.

We value all contributions, including:
*   **Feature suggestions or bug reports:** [Go to Issues page](https://github.com/FinStep-AI/ContestTrade/issues)
*   **Testing feedback:** Test results, user experience, etc.

## Star History

<div align="center">
  <a href="https://star-history.com/#FinStep-AI/ContestTrade&Date">
    <img src="https://api.star-history.com/svg?repos=FinStep-AI/ContestTrade&type=Date" alt="Star History Chart" style="width: 80%;">
  </a>
</div>

## Disclaimer

**Important Disclaimer:** The `ContestTrade` project is an open-source quantitative trading research framework, only for academic and educational uses. The project's examples, data, and analysis results do not offer any investment advice.

**Risk Warning:**

*   **Market Risk:** This project does not provide investment, financial, legal, or tax advice. Trading signals and analyses come from AI model deductions based on historical data and should not be used as the basis for buy or sell actions.
*   **Data Accuracy:** Framework data sources might be delayed, inaccurate, or incomplete. Data reliability is not guaranteed.
*   **Model Hallucination:** AI models (including Large Language Models) have limitations and may "hallucinate." Information generated by the framework is not guaranteed for accuracy, completeness, or timeliness.
*   **Liability:** The developers accept no liability for any direct or indirect losses from using or being unable to use this framework. Investing includes risks; trade carefully.

**Understand the risks before using this framework for actual trading decisions.**

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