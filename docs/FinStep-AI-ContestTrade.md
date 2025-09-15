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

# ContestTrade: Automated Multi-Agent Trading System

**Revolutionize your trading strategy with ContestTrade, an innovative multi-agent framework designed to autonomously identify and capitalize on event-driven investment opportunities.** ([Original Repository](https://github.com/FinStep-AI/ContestTrade))

## Key Features

*   **Automated Stock Selection:**  Automated scanning of the entire market to generate a list of tradeable stock candidates without manual screening.
*   **Event-Driven Strategy:**  Focuses on opportunities triggered by significant events like news, announcements, and policy changes, emphasizing high-impact information.
*   **Customizable Configuration:** Supports user-defined agent preferences and trading strategies, enabling flexibility for various investment styles.
*   **Multi-Agent Architecture:** Leverages a two-stage "contest" pipeline to refine decisions and ensure robustness.

## Introduction

ContestTrade is a cutting-edge multi-agent trading framework specifically designed for event-driven stock selection. It automatically discovers, evaluates, and tracks event-driven investment opportunities without human intervention. The system delivers executable asset allocation recommendations, streamlining the investment process.

## Framework Overview

<p align="center">
  <img src="assets/architecture.jpg" style="width: 90%; height: auto;">
</p>

The ContestTrade framework operates via a structured two-stage pipeline, simulating an investment firm's decision-making process. This "dual-contest" approach ensures final decisions are based on the most reliable insights, offering strong adaptability in complex market conditions.

1.  **Data Processing Stage:** The Data Team receives raw market data, which multiple Data Analysis Agents process into structured "textual factors." An internal contest evaluates the potential value of these factors, creating an optimal "factor portfolio."

2.  **Research and Decision-Making Stage:** The Research Team receives the factor portfolio. Multiple Research Agents analyze the factors using their unique "Trading Beliefs" and financial tools, submitting trading proposals. A second internal contest synthesizes these proposals into a unified, reliable asset allocation strategy.

## Installation

Follow these steps to get ContestTrade up and running:

```bash
# 1. Clone the repository
git clone https://github.com/FinStep-AI/ContestTrade.git
cd ContestTrade

# 2. (Recommended) Create and activate a virtual environment
conda create -n contesttrade python=3.10
conda activate contesttrade

# 3. Install dependencies
pip install -r requirements.txt
```

Alternatively, deploy with [Docker](https://docs.n8n.io/hosting/installation/docker/):

```
docker run -it --rm --name contest_trade -v $(pwd)/config.yaml:/ContestTrade/config.yaml finstep/contesttrade:v2.0
```

## Configuration

Before using ContestTrade, you'll need to configure API keys and LLM parameters.

Edit the `config_us.yaml` file and enter your API keys:

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

> Note: You'll need to obtain your own LLM and VLM API keys.  Enter the URL, API Key, and model name based on the platform and model you are using.

## Preference Settings

Each Research Agent utilizes a "trading belief" to generate investment signals based on data and tools. Configure these beliefs in `contest_trade/config/belief_list.json`.

*Example 1: Short-Term Focus*

```json
[
  "Focus on short-term event-driven opportunities: prioritize company announcements, M&A and restructuring, sudden order increases, technological breakthroughs and other catalysts; prefer mid/small-cap, high-volatility thematic stocks, suitable for aggressive arbitrage strategies."
]
```

*Example 2: Stable Events Focus*

```json
[
  "Focus on stable, high-certainty events: pay attention to dividends, buybacks, earnings forecast confirmations, major contract signings and policy tailwinds; prefer large-cap blue-chips with low volatility and high certainty, suitable for conservative allocation."
]
```

*Default Configuration:*

```json
[
  "Based on the provided information, comprehensively consider each company's business dynamics, industry trends and potential market impact. Recommend stock portfolios with short-term investment potential for two groups: Group 1 — risk-seekers (prefer high volatility, high returns, mid/low market cap stocks); Group 2 — conservative investors (prefer low volatility, stable returns, large-cap stocks).",
  "Based on recent sudden events, policy adjustments and company announcements as catalysts, combined with market sentiment transmission paths and capital game features. Screen event-driven opportunities for two different styles: Group 1 — aggressive arbitrage (prefer restructuring expectation, sudden order increases, technical breakthroughs in small-cap stocks); Group 2 — defensive arbitrage (prefer dividend increases, large buybacks, acquisition of franchise rights in blue-chip stocks). Pay attention to northbound capital movement and institutional seat trends on the trading leaderboard for resonance effects."
]
```

## Usage

Easily run ContestTrade using the Command Line Interface (CLI):

```bash
python -m cli.main run
```

The CLI provides an interactive interface to select the market for analysis.

<p align="center">
  <img src="assets/contest_trade_cli_select_market.jpg" style="width: 100%; height: auto;">
</p>

View agent signals after completion.

<p align="center">
  <img src="assets/contest_trade_cli_main_us.jpg" style="width: 100%; height: auto;">
</p>

Access detailed research reports.

<p align="center">
  <img src="assets/contest_trade_cli_report_us.jpg" style="width: 100%; height: auto;">
</p>

View detailed research reports.

<p align="center">
  <img src="assets/contest_trade_cli_research_report_us.jpg" style="width: 100%; height: auto;">
</p>

Access detailed data analysis reports.

<p align="center">
  <img src="assets/contest_trade_cli_data_report_us.jpg" style="width: 100%; height: auto;">
</p>

> These reports are saved in Markdown format within the `contest_trade/agents_workspace/results` directory.

## 🌟 Vision & Roadmap

Our vision is to contribute to the era of AGI by leveraging open-source and exploring new paradigms in quantitative trading.

Our goals are to develop more robust infrastructure, enhance agent diversity, explore the boundaries of AI in financial trading, and create a reliable and scalable agent trading framework.

### Roadmap

*   **V1.1 (Finished): Framework Stability Enhancement & Core Experience Optimization**
    *   [✓] Decoupled core data source module for multiple data source adaptors. (`data-provider` refactor)
    *   [✓] Improved CLI logging and user interaction.

*   **V2.0 (Finished): Market and Function Expansion**
    *   [✓] US stock market data access.
    *   [✓] Enhanced factors and signal sources.

*   **Future Plans:**
    *   [ ] Support for Hong Kong and other stock markets.
    *   [ ] Visual backtesting and analysis interface.
    *   [ ] Scalability for increased agents.

## Contributing

ContestTrade is a community-driven open-source project!

If you're a developer, see our **[Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md)**.

We also appreciate non-code contributions, including:

*   Proposing new features or reporting bugs via the [Issues page](https://github.com/FinStep-AI/ContestTrade/issues).
*   Providing feedback on testing results and user experiences.

## Star History

<div align="center">
  <a href="https://star-history.com/#FinStep-AI/ContestTrade&Date">
    <img src="https://api.star-history.com/svg?repos=FinStep-AI/ContestTrade&type=Date" alt="Star History Chart" style="width: 80%;">
  </a>
</div>

## Disclaimer

**Important Disclaimer:** `ContestTrade` is an open-source research framework for academic and educational use only. Any examples, data, and analysis results are not investment advice.

**Risk Warning:**

*   **Market Risk:** This project is for research, not investment, financial, legal, or tax advice. AI-generated outputs are based on historical data and should not be used for buying or selling decisions.
*   **Data Accuracy:** The framework uses data sources potentially subject to delays or inaccuracies. We don't guarantee data reliability.
*   **Model Hallucination:** AI models may have limitations, including "hallucination." We don't guarantee the information's accuracy, completeness, or timeliness.
*   **Liability:** Developers are not liable for direct or indirect losses from using this framework. Investing involves risks; use caution.

**Thoroughly understand all risks before using this framework for any actual trading decisions.**

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