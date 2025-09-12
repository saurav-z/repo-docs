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

# ContestTrade: Revolutionizing AI-Driven Trading with a Multi-Agent Framework

**ContestTrade** is a cutting-edge multi-agent trading framework designed to autonomously identify, evaluate, and capitalize on event-driven investment opportunities.  Explore the original repo on [GitHub](https://github.com/FinStep-AI/ContestTrade) for more details.

## Key Features

*   **Automated Stock Selection:** Scans the entire market to generate tradable stock lists, eliminating the need for manual screening.
*   **Event-Driven Strategy:** Focuses on opportunities triggered by catalysts like news, announcements, and policy changes for significant information impact.
*   **Customizable Configurations:** Enables users to tailor agent research preferences and strategies, adapting to diverse investment styles.
*   **Multi-Agent Architecture:** Leverages a two-stage pipeline with data and research teams, mimicking an investment firm's decision-making process.
*   **US Stock Market Support:** Analyzes data from the US stock market, providing actionable insights.
*   **Open Source & Community Driven:** Built on open-source principles, we welcome and encourage contributions.

## Introduction

ContestTrade offers a unique approach to automated trading, focusing on event-driven stock selection within a multi-agent framework.  The system autonomously discovers, assesses, and tracks event-driven opportunities with the goal of generating actionable asset allocation recommendations without human intervention. The framework is designed to mimic the workflow of a financial firm's investment process.

## Framework Overview

<p align="center">
  <img src="assets/architecture.jpg" style="width: 90%; height: auto;">
</p>

ContestTrade operates through a two-stage pipeline designed to mimic the decision-making process of an investment firm.

1.  **Data Processing Stage:** The Data Team receives raw market data from various sources. Data Analysis Agents process the data into structured "textual factors," and the internal contest mechanism assesses the value of each agent's factors to construct an optimal "factor portfolio."

2.  **Research and Decision-Making Stage:** The Research Team receives the optimal factor portfolio. Research Agents, each with unique "Trading Beliefs" and access to financial tools, analyze the factors and submit trading proposals. A second internal contest evaluates the proposals, resulting in a unified asset allocation strategy.

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

Before running ContestTrade, you need to configure API keys and LLM parameters.

Edit `config_us.yaml` with your API keys. Here's a table of required and optional keys:

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

> Note: You must obtain the LLM and VLM API keys and input the correct model names, base URLs, and API keys.

## Preference

Each Research Agent uses a "trading belief." Investment signals are generated based on these beliefs in combination with data and tools (up to 5 signals per belief). The configuration file, `contest_trade/config/belief_list.json`, is formatted as a JSON array of strings.

Example 1 — Preference for Short-Term Event-Driven (More Aggressive):

```json
[
  "Focus on short-term event-driven opportunities: prioritize company announcements, M&A and restructuring, sudden order increases, technological breakthroughs and other catalysts; prefer mid/small-cap, high-volatility thematic stocks, suitable for aggressive arbitrage strategies."
]
```

Example 2 — Preference for Stable Events (More Conservative):

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

Start ContestTrade using the Command Line Interface (CLI).

```bash
python -m cli.main run
```

After startup, you can choose the market to analyze via the terminal interface. The default analysis time is the current time.

<p align="center">
  <img src="assets/contest_trade_cli_select_market.jpg" style="width: 100%; height: auto;">
</p>

View the agent signals after all agents have finished.

<p align="center">
  <img src="assets/contest_trade_cli_main_us.jpg" style="width: 100%; height: auto;">
</p>

View more detailed reports.

<p align="center">
  <img src="assets/contest_trade_cli_report_us.jpg" style="width: 100%; height: auto;">
</p>

<p align="center">
  <img src="assets/contest_trade_cli_research_report_us.jpg" style="width: 100%; height: auto;">
</p>

<p align="center">
  <img src="assets/contest_trade_cli_data_report_us.jpg" style="width: 100%; height: auto;">
</p>

> Reports will be saved in Markdown format in the `contest_trade/agents_workspace/results` directory.

## 🌟 Vision & Roadmap

We are committed to leveraging the power of the open-source community to discover new paradigms of quantitative trading in the era of AGI.

Our focus is on developing a robust infrastructure, expanding the variety of agents, and exploring the capabilities of AI in financial trading to create a stable, reliable, and scalable agent-based trading framework.

### Roadmap

**V1.1 (Finished): Framework Stability Enhancement & Core Experience Optimization**

*   [✓] Decoupled the core data source module to support multiple data sources. (`data-provider` refactor)
*   [✓] Optimized CLI logging and user interaction.

**V2.0 (Finished): Market and Function Expansion**

*   [✓] Access to **US stock** market data.
*   [✓] Enhanced the factors and signal sources.

**Future Plans:**

*   [ ] Support for Hong Kong and other stock markets
*   [ ] Visual backtesting and analysis interface
*   [ ] Support scaling up more agents

## Contributing

ContestTrade thrives on community contributions!

Refer to our **[Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md)** to become a contributor.

Non-code contributions are also valued:

*   Proposing feature suggestions or reporting bugs: [Go to Issues page](https://github.com/FinStep-AI/ContestTrade/issues)
*   Providing feedback on your testing results: Including test results, user experience, etc.

## Star History

<div align="center">
  <a href="https://star-history.com/#FinStep-AI/ContestTrade&Date">
    <img src="https://api.star-history.com/svg?repos=FinStep-AI/ContestTrade&type=Date" alt="Star History Chart" style="width: 80%;">
  </a>
</div>

## Disclaimer

**Important Disclaimer:** ContestTrade is an open-source quantitative trading research framework intended for academic and educational purposes only. The outputs, including trading signals and analyses, do not constitute any form of investment advice.

**Risk Warning:**

*   **Market Risk:** This project does not provide investment, financial, legal, or tax advice. The outputs are the results of AI model inferences and should not be used for buy/sell operations.
*   **Data Accuracy:** Data sources may have delays, inaccuracies, or incompleteness. The reliability of the data is not guaranteed.
*   **Model Hallucination:** AI models have inherent limitations, and "hallucination" is possible. The accuracy, completeness, and timeliness of the information generated by the framework are not guaranteed.
*   **Liability:** Developers are not liable for any direct or indirect losses from using or being unable to use this framework. Investing carries risks; proceed with caution.

**Thoroughly understand all associated risks before using this framework for trading decisions.**

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