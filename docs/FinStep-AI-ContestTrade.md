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

# ContestTrade: An AI-Powered Multi-Agent Trading System

**ContestTrade** is a cutting-edge multi-agent trading framework that harnesses the power of AI to autonomously identify and capitalize on event-driven investment opportunities, providing data-driven insights without human intervention.  [Explore the original repo on GitHub!](https://github.com/FinStep-AI/ContestTrade)

## Key Features

*   **Automated Stock Selection:** Scans the entire market and generates tradable stock lists automatically.
*   **Event-Driven Strategy:** Focuses on opportunities triggered by catalysts like news, announcements, and policy changes.
*   **Multi-Agent Architecture:** Employs a two-stage pipeline with data and research teams for robust decision-making.
*   **Personalized Configuration:** Supports user-defined preferences and strategies to adapt to various investment styles.

## Introduction

ContestTrade is designed to automatically discover, evaluate, and track event-driven opportunities, generating actionable asset allocation recommendations. It utilizes a multi-agent system to simulate the decision-making process of an investment firm. This framework allows for automated market analysis and strategic investment choices.

## Framework Overview

The ContestTrade workflow operates in two main stages:

1.  **Data Processing Stage:**  Raw market data from various sources is processed by the **Data Team**, which comprises multiple Data Analysis Agents.  These agents refine the raw data into structured "textual factors," and an internal contest mechanism identifies the most valuable factors.

2.  **Research and Decision-Making Stage:**  The optimal factor portfolio from the Data Team is then passed to the **Research Team**. Multiple Research Agents, each with unique "Trading Beliefs" and access to financial tools, conduct in-depth analyses.  A second internal contest evaluates these proposals, resulting in a consolidated asset allocation strategy.

<p align="center">
  <img src="assets/architecture.jpg" style="width: 90%; height: auto;">
</p>

## Installation

Get started with ContestTrade by following these steps:

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

Alternatively, you can deploy using [Docker](https://docs.n8n.io/hosting/installation/docker/):

```
docker run -it --rm --name contest_trade -v $(pwd)/config.yaml:/ContestTrade/config.yaml finstep/contesttrade:v2.0
```

## Configuration

Configure the necessary API keys and LLM parameters in the `config_us.yaml` file.

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

> Note: Obtain and configure your LLM and VLM API keys based on your chosen platform and model.

## Preference Settings

Customize the trading behavior of each Research Agent through "trading beliefs," defined in the `contest_trade/config/belief_list.json` file.

*Example preferences for short-term event-driven strategies:*

```json
[
  "Focus on short-term event-driven opportunities: prioritize company announcements, M&A and restructuring, sudden order increases, technological breakthroughs and other catalysts; prefer mid/small-cap, high-volatility thematic stocks, suitable for aggressive arbitrage strategies."
]
```

*Example preferences for stable events:*

```json
[
  "Focus on stable, high-certainty events: pay attention to dividends, buybacks, earnings forecast confirmations, major contract signings and policy tailwinds; prefer large-cap blue-chips with low volatility and high certainty, suitable for conservative allocation."
]
```

*Default configuration:*

```json
[
  "Based on the provided information, comprehensively consider each company's business dynamics, industry trends and potential market impact. Recommend stock portfolios with short-term investment potential for two groups: Group 1 — risk-seekers (prefer high volatility, high returns, mid/low market cap stocks); Group 2 — conservative investors (prefer low volatility, stable returns, large-cap stocks).",
  "Based on recent sudden events, policy adjustments and company announcements as catalysts, combined with market sentiment transmission paths and capital game features. Screen event-driven opportunities for two different styles: Group 1 — aggressive arbitrage (prefer restructuring expectation, sudden order increases, technical breakthroughs in small-cap stocks); Group 2 — defensive arbitrage (prefer dividend increases, large buybacks, acquisition of franchise rights in blue-chip stocks). Pay attention to northbound capital movement and institutional seat trends on the trading leaderboard for resonance effects."
]
```

## Usage

Run ContestTrade via the Command Line Interface (CLI):

```bash
python -m cli.main run
```

Follow the on-screen prompts to select your market and view the analysis results, including detailed research and data reports.

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

> All reports are saved in the `contest_trade/agents_workspace/results` directory in Markdown format for easy reference and sharing.

## 🌟 Vision & Roadmap

ContestTrade is dedicated to exploring the future of quantitative trading in the AGI era.

### Roadmap

**V1.1 (Completed): Framework Stability Enhancement & Core Experience Optimization**

*   [✓] Data source module decoupling for multiple data source support.
*   [✓] Optimized CLI logging and user interaction.

**V2.0 (Completed): Market and Function Expansion**

*   [✓] US stock market data integration.
*   [✓] Introduction of richer factors and signal sources.

**Future Plans:**

*   [ ] Support for Hong Kong and other stock markets.
*   [ ] Visual backtesting and analysis interface.
*   [ ] Scalability for more agents.

## Contributing

Join the ContestTrade community!  Contributions are welcome in all forms.  See the **[Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md)**.

*   **Report issues and suggest features:** [Go to Issues page](https://github.com/FinStep-AI/ContestTrade/issues)
*   **Provide feedback:** Share your testing results and user experience.

## Star History

<div align="center">
  <a href="https://star-history.com/#FinStep-AI/ContestTrade&Date">
    <img src="https://api.star-history.com/svg?repos=FinStep-AI/ContestTrade&type=Date" alt="Star History Chart" style="width: 80%;">
  </a>
</div>

## Disclaimer

**Important:** This project is for research and educational purposes.  It does not constitute investment advice.

**Risk Warning:**

*   **Market Risk:** The project's outputs are based on historical data and AI model deductions and should not be used as the basis for any investment decisions.
*   **Data Accuracy:** Data sources may be subject to delays, inaccuracies, or incompleteness; the reliability of the data is not guaranteed.
*   **Model Limitations:** AI models can produce inaccurate information ("hallucination").  We do not guarantee the accuracy of the information generated.
*   **Liability:** The developers are not liable for any losses resulting from the use of this framework.  Investing involves risk; exercise caution.

**Always fully understand the risks before using this framework for actual trading.**

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