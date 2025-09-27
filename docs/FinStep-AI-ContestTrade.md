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
# ContestTrade: Automated AI-Driven Multi-Agent Trading Framework

**ContestTrade** is a cutting-edge, open-source framework that utilizes a multi-agent system and internal contest mechanisms to automatically identify and capitalize on event-driven trading opportunities. Explore the original repository: [https://github.com/FinStep-AI/ContestTrade](https://github.com/FinStep-AI/ContestTrade).

## Key Features

*   **Automated Stock Selection:**  Automatically scans the entire market to identify potential investment opportunities, eliminating the need for manual screening.
*   **Event-Driven Strategy:** Focuses on catalysts like news, announcements, capital flows, and policy changes to pinpoint high-impact investment events.
*   **Multi-Agent Architecture:** Leverages a team of specialized agents for data processing, research, and decision-making, mimicking a professional investment firm.
*   **Customizable Configuration:** Allows users to tailor agent strategies and preferences to align with specific investment styles and risk profiles.
*   **Multi-Market Support:** Supports US stock market data (with plans for Hong Kong and other markets).
*   **CLI Interface:** Provides a user-friendly command-line interface for easy operation and result visualization.

## Introduction

ContestTrade is a sophisticated multi-agent trading framework designed to automate the discovery, evaluation, and tracking of event-driven opportunities in the stock market. The system aims to provide actionable asset allocation recommendations without human intervention. It achieves this through a two-stage pipeline, the Data Processing Stage and Research and Decision-Making Stage.

## Framework Overview

<p align="center">
  <img src="assets/architecture.jpg" style="width: 90%; height: auto;">
</p>

ContestTrade's workflow operates in two primary stages:

1.  **Data Processing Stage:** Raw market data is fed to the **Data Team**. Data Analysis Agents work in parallel to create structured "textual factors." The internal contest mechanism evaluates these factors, building an optimal "factor portfolio."
2.  **Research and Decision-Making Stage:** The factor portfolio is passed to the **Research Team**. Research Agents analyze factors and submit trading proposals. A second contest evaluates the proposals, producing a unified asset allocation strategy.

## Installation

To get started with ContestTrade, follow these steps:

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

Configure API keys and LLM parameters by editing the `config_us.yaml` file. Here's a table of required and optional keys:

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

> Note:  LLM and VLM API keys and URLs need to be applied for separately.

## Preference

Each Research Agent corresponds to a "trading belief," determining its investment signals. Configure these beliefs in `contest_trade/config/belief_list.json`.

Example short-term event-driven preference:

```json
[
  "Focus on short-term event-driven opportunities: prioritize company announcements, M&A and restructuring, sudden order increases, technological breakthroughs and other catalysts; prefer mid/small-cap, high-volatility thematic stocks, suitable for aggressive arbitrage strategies."
]
```

Example conservative preference:

```json
[
  "Focus on stable, high-certainty events: pay attention to dividends, buybacks, earnings forecast confirmations, major contract signings and policy tailwinds; prefer large-cap blue-chips with low volatility and high certainty, suitable for conservative allocation."
]
```

## Usage

Run ContestTrade using the Command Line Interface (CLI):

```bash
python -m cli.main run
```

Follow the prompts to select the market and view the results, including detailed research and data analysis reports, which are saved in the `contest_trade/agents_workspace/results` directory.

## 🌟 Vision & Roadmap

The project aims to explore new quantitative trading paradigms in the AGI era.

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

Contribute to ContestTrade! Review the **[Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md)** for developer guidelines.  Non-code contributions, such as feature suggestions and bug reports, are also welcome via the [Issues page](https://github.com/FinStep-AI/ContestTrade/issues).

## Star History

<div align="center">
  <a href="https://star-history.com/#FinStep-AI/ContestTrade&Date">
    <img src="https://api.star-history.com/svg?repos=FinStep-AI/ContestTrade&type=Date" alt="Star History Chart" style="width: 80%;">
  </a>
</div>

## Disclaimer

**Important Disclaimer:** This project, `ContestTrade`, is an open-source quantitative trading research framework intended for academic and educational purposes only. The examples, data, and analysis results included in the project do not constitute any form of investment advice.

**Risk Warning:**
*   **Market Risk:** This project does not constitute any form of investment, financial, legal, or tax advice. All outputs, including trading signals and analyses, are the results of AI model deductions based on historical data and should not be considered a basis for any buy or sell operations.
*   **Data Accuracy:** The data sources used by the framework may be subject to delays, inaccuracies, or incompleteness. We do not guarantee the reliability of the data.
*   **Model Hallucination:** AI models (including Large Language Models) have inherent limitations and are subject to the risk of "hallucination." We do not guarantee the accuracy, completeness, or timeliness of the information generated by the framework.
*   **Liability:** The developers assume no liability for any direct or indirect losses resulting from the use of or inability to use this framework. Investing involves risks; enter the market with caution.

**Before using this framework for any actual trading decisions, be sure to fully understand the associated risks.**

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

This project is licensed under the [Apache 2.0 License](LICENSE).