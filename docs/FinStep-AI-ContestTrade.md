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

# ContestTrade: An AI-Powered Multi-Agent Trading Framework

**ContestTrade** is an innovative multi-agent system designed to automate and optimize event-driven trading strategies.  [Explore the code on GitHub](https://github.com/FinStep-AI/ContestTrade).

## Key Features

*   **Automated Stock Selection:** Scans the entire market and generates tradable stock lists without manual intervention.
*   **Event-Driven Strategy:**  Focuses on opportunities triggered by news, announcements, capital flows, and policy changes.
*   **Customizable Agents:** Supports user-defined research preferences and trading strategies.
*   **Multi-Agent Framework:** Employs a two-stage pipeline with internal contests to ensure robust and effective trading decisions.
*   **US Stock Market Support:** Provides access to US stock market data.
*   **CLI Interface:** Offers a command-line interface for easy use and report viewing.

## Introduction

ContestTrade is a pioneering multi-agent trading framework that leverages AI to identify and capitalize on event-driven investment opportunities.  The system autonomously discovers, evaluates, and tracks event-driven opportunities, providing actionable asset allocation recommendations.

## Framework Overview

<p align="center">
  <img src="assets/architecture.jpg" style="width: 90%; height: auto;">
</p>

ContestTrade operates through a two-stage pipeline, simulating the decision-making process of an investment firm:

1.  **Data Processing Stage:**  Raw market data is processed by the **Data Team**, with multiple Data Analysis Agents generating "textual factors." An internal contest mechanism then constructs an optimal "factor portfolio."
2.  **Research and Decision-Making Stage:** This optimal factor portfolio is analyzed by the **Research Team**. Research Agents, each with unique "Trading Beliefs," conduct in-depth analyses and propose trading strategies. A second round of internal contests synthesizes these proposals into a unified asset allocation strategy.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/FinStep-AI/ContestTrade.git
    cd ContestTrade
    ```
2.  **(Recommended) Create and Activate a Virtual Environment:**
    ```bash
    conda create -n contesttrade python=3.10
    conda activate contesttrade
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

**Or, deploy with Docker:**

```bash
docker run -it --rm --name contest_trade -v $(pwd)/config.yaml:/ContestTrade/config.yaml finstep/contesttrade:v1.1
```

## Configuration

Configure API keys and LLM parameters in `config_us.yaml`.  Required keys are marked below:

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

> *Note:*  You must obtain your own LLM and VLM API keys.

## Preference: Trading Beliefs Configuration

Customize agent behavior by configuring "trading beliefs" in `contest_trade/config/belief_list.json`.  This file is a JSON array of strings.

**Example: Short-term Event-Driven (Aggressive)**

```json
[
  "Focus on short-term event-driven opportunities: prioritize company announcements, M&A and restructuring, sudden order increases, technological breakthroughs and other catalysts; prefer mid/small-cap, high-volatility thematic stocks, suitable for aggressive arbitrage strategies."
]
```

**Example: Stable Events (Conservative)**

```json
[
  "Focus on stable, high-certainty events: pay attention to dividends, buybacks, earnings forecast confirmations, major contract signings and policy tailwinds; prefer large-cap blue-chips with low volatility and high certainty, suitable for conservative allocation."
]
```

**Default Configuration:**  (Included in original README)

## Usage

Run ContestTrade using the CLI:

```bash
python -m cli.main run
```

Follow the interactive terminal prompts to select a market and analyze.  Reports are saved in the `contest_trade/agents_workspace/results` directory in Markdown format.  (Images of CLI interaction, results, and reports included).

## 🌟 Vision & Roadmap

We are dedicated to advancing quantitative trading in the AGI era, with a focus on developing a robust infrastructure and expanding the capabilities of our agents.

### Roadmap

*   **V1.1 (Completed):** Framework Stability Enhancement & Core Experience Optimization
*   **V2.0 (Completed):** Market and Function Expansion (US Stocks, richer factors).
*   **Future Plans:**
    *   Support for Hong Kong and other markets.
    *   Visual backtesting and analysis interface.
    *   Support for scaling up more agents.

## Contributing

We encourage contributions! See the **[Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md)** for development guidance.

Non-code contributions, such as feature suggestions and bug reports, are also welcome via the [Issues page](https://github.com/FinStep-AI/ContestTrade/issues).

## Star History

(Star History chart included.)

## Disclaimer

**Important Disclaimer:** This project is for academic and educational purposes only and does not constitute investment advice.

**Risk Warning:**
*   Market Risk
*   Data Accuracy
*   Model Hallucination
*   Liability

**Before using this framework for any actual trading decisions, be sure to fully understand the associated risks.**

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

Licensed under the [Apache 2.0 License](LICENSE).