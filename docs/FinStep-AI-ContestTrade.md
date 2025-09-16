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
# ContestTrade: Your AI-Powered Advantage in Event-Driven Trading

**ContestTrade** is a cutting-edge, multi-agent trading framework designed to autonomously identify and capitalize on event-driven investment opportunities, all without human intervention. ([View the source code on GitHub](https://github.com/FinStep-AI/ContestTrade))

## Key Features

*   **Automated Stock Selection:**  Scans the entire market and generates tradable stock lists based on potential events.
*   **Event-Driven Focus:** Prioritizes opportunities triggered by catalysts like news, announcements, and policy changes.
*   **Customizable Strategy:** Supports user-defined agent preferences and strategies for flexible adaptation to different investment styles.
*   **Multi-Agent Architecture:** Utilizes a two-stage contest framework, mimicking investment firm workflows, for robust decision-making.
*   **Market Expansion:** Includes support for US stock market data (V2.0).

## Overview: The ContestTrade Framework

<p align="center">
  <img src="assets/architecture.jpg" style="width: 90%; height: auto;">
</p>

ContestTrade's workflow is a structured two-stage pipeline, mimicking the dynamic decision-making process of an investment firm. This dual-contest framework ensures that final decisions are driven only by the most robust and effective insights, maintaining strong adaptability and resilience in complex markets.

1.  **Data Processing Stage:**  The **Data Team** receives raw market data from various sources. Multiple Data Analysis Agents process this data into structured "textual factors." An internal contest mechanism then evaluates the factors, resulting in an optimal "factor portfolio."

2.  **Research and Decision-Making Stage:** The factor portfolio is passed to the **Research Team**. Research Agents, each with their "Trading Beliefs," conduct analyses and submit trading proposals. A second contest evaluates these proposals, generating a unified asset allocation strategy.

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

Alternatively, deploy with [Docker](https://docs.n8n.io/hosting/installation/docker/):

```bash
docker run -it --rm --name contest_trade -v $(pwd)/config.yaml:/ContestTrade/config.yaml finstep/contesttrade:v2.0
```

## Configuration: Setting Up Your Trading Environment

Configure your trading environment by editing the `config_us.yaml` file with your API keys:

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

> *Note:* You must obtain your own LLM and VLM API keys.  Enter the URL, API Key, and model name based on your chosen platform.

## Preference: Defining Your Trading Beliefs

Customize agent behavior through the `contest_trade/config/belief_list.json` file. This file uses a JSON array of strings to define trading preferences.

**Example 1: Short-Term, Aggressive Approach**

```json
[
  "Focus on short-term event-driven opportunities: prioritize company announcements, M&A and restructuring, sudden order increases, technological breakthroughs and other catalysts; prefer mid/small-cap, high-volatility thematic stocks, suitable for aggressive arbitrage strategies."
]
```

**Example 2: Stable, Conservative Approach**

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

## Usage: Running ContestTrade

Use the Command Line Interface (CLI) to launch the system:

```bash
python -m cli.main run
```

The CLI provides an interactive interface to select the market and analyze data.

<p align="center">
  <img src="assets/contest_trade_cli_select_market.jpg" style="width: 100%; height: auto;">
</p>

View agent signals in the results summary:

<p align="center">
  <img src="assets/contest_trade_cli_main_us.jpg" style="width: 100%; height: auto;">
</p>

Access detailed research reports:

<p align="center">
  <img src="assets/contest_trade_cli_report_us.jpg" style="width: 100%; height: auto;">
</p>

<p align="center">
  <img src="assets/contest_trade_cli_research_report_us.jpg" style="width: 100%; height: auto;">
</p>

And detailed data analysis reports:

<p align="center">
  <img src="assets/contest_trade_cli_data_report_us.jpg" style="width: 100%; height: auto;">
</p>

> *Note:* Reports are saved in Markdown format within the `contest_trade/agents_workspace/results` directory.

## 🌟 Vision & Roadmap

We are committed to advancing quantitative trading in the era of AGI, supported by the open-source community.

The project focuses on expanding infrastructure, agent variety, and exploring AI's potential in financial trading.

### Roadmap

**V1.1 (Finished): Framework Enhancement & Experience Optimization**
- [✓] Decoupled data source module (`data-provider` refactor).
- [✓] Improved CLI logging and interaction.

**V2.0 (Finished): Market and Function Expansion**
- [✓] Added US stock market data.
- [✓] Introduced richer factors and signal sources.

**Future Plans:**
- [ ] Support for Hong Kong and other stock markets.
- [ ] Visual backtesting and analysis interface.
- [ ] Expand the number of agents.

## Contributing

Contribute to ContestTrade!  We welcome all contributions:

*   **Code contributions:** Refer to the **[Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md)**.
*   **Non-code contributions:**  Provide feature suggestions, report bugs on the [Issues page](https://github.com/FinStep-AI/ContestTrade/issues), and offer feedback on your testing experiences.

## Star History

<div align="center">
  <a href="https://star-history.com/#FinStep-AI/ContestTrade&Date">
    <img src="https://api.star-history.com/svg?repos=FinStep-AI/ContestTrade&type=Date" alt="Star History Chart" style="width: 80%;">
  </a>
</div>

## Disclaimer

**Important:  This project is for research and educational purposes only. It does not provide financial advice.**

**Risk Warning:**

*   **Market Risk:**  Outputs, signals, and analyses are based on AI model deductions and are not investment advice.  Use these as a basis for buy/sell operations at your own risk.
*   **Data Accuracy:** Data sources may have delays, inaccuracies, or omissions.  We do not guarantee data reliability.
*   **Model Hallucination:** AI models have limitations and may generate inaccurate information. We do not guarantee the accuracy, completeness, or timeliness of information.
*   **Liability:**  Developers are not liable for losses from the use of this framework.  Investing involves risks; proceed with caution.

**Before using this framework for any trading decisions, fully understand the risks involved.**

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