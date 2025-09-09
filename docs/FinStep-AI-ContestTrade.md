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
# ContestTrade: AI-Powered Multi-Agent Trading Framework

**Unleash the power of AI in your trading strategy with ContestTrade, an innovative multi-agent framework designed to automate event-driven stock selection.** [Explore the original repository](https://github.com/FinStep-AI/ContestTrade)

## Key Features

*   **Automated Stock Selection:** Scan the entire market and generate tradable stock lists automatically, eliminating manual screening.
*   **Event-Driven Analysis:** Focus on opportunities triggered by news, announcements, capital flows, and policy changes.
*   **Customizable Configuration:** Tailor agent research preferences and strategies to align with diverse investment styles.
*   **Multi-Agent Architecture:** Simulate a dynamic investment firm with data processing and research teams working in parallel, with an internal contest mechanism.
*   **Multi-Market Support:** Access US stock market data.

## Introduction

ContestTrade is a multi-agent trading framework meticulously designed for event-driven stock selection. Its primary objective is to autonomously identify, evaluate, and track event-driven investment opportunities, ultimately delivering actionable asset allocation recommendations without human intervention. The system's architecture simulates the workflows of an investment firm, fostering adaptability and resilience in complex market environments.

## Framework Overview

<p align="center">
  <img src="assets/architecture.jpg" style="width: 90%; height: auto;">
</p>

ContestTrade operates through a structured two-stage pipeline, mirroring the decision-making processes of an investment firm. This dual-contest framework ensures that final decisions are driven by the most robust and effective insights.

1.  **Data Processing Stage:** Raw market data is processed by the **Data Team**. Multiple Data Analysis Agents transform raw data into structured "textual factors". An internal contest mechanism evaluates the potential value of each factor, creating an optimal "factor portfolio."

2.  **Research and Decision-Making Stage:** The factor portfolio is passed to the **Research Team**. Research Agents, each with unique "Trading Beliefs" and access to financial tools, conduct in-depth analyses and propose trading strategies. A second contest evaluates the proposals, culminating in a unified asset allocation strategy.

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

**Alternative Installation with Docker:**

```bash
docker run -it --rm --name contest_trade -v $(pwd)/config.yaml:/ContestTrade/config.yaml finstep/contesttrade:v1.1
```

## Configuration

Before running ContestTrade, configure API keys and LLM parameters in `config_us.yaml`.

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

> Note: Obtain LLM and VLM API keys and provide the corresponding URL, API Key, and model name.

## Preference Configuration

Customize Research Agent behavior using "trading beliefs" in `contest_trade/config/belief_list.json`.

*   **Example 1 (Aggressive):**
    ```json
    [
      "Focus on short-term event-driven opportunities: prioritize company announcements, M&A and restructuring, sudden order increases, technological breakthroughs and other catalysts; prefer mid/small-cap, high-volatility thematic stocks, suitable for aggressive arbitrage strategies."
    ]
    ```

*   **Example 2 (Conservative):**
    ```json
    [
      "Focus on stable, high-certainty events: pay attention to dividends, buybacks, earnings forecast confirmations, major contract signings and policy tailwinds; prefer large-cap blue-chips with low volatility and high certainty, suitable for conservative allocation."
    ]
    ```

*   **Default Configuration:**
    ```json
    [
      "Based on the provided information, comprehensively consider each company's business dynamics, industry trends and potential market impact. Recommend stock portfolios with short-term investment potential for two groups: Group 1 — risk-seekers (prefer high volatility, high returns, mid/low market cap stocks); Group 2 — conservative investors (prefer low volatility, stable returns, large-cap stocks).",
      "Based on recent sudden events, policy adjustments and company announcements as catalysts, combined with market sentiment transmission paths and capital game features. Screen event-driven opportunities for two different styles: Group 1 — aggressive arbitrage (prefer restructuring expectation, sudden order increases, technical breakthroughs in small-cap stocks); Group 2 — defensive arbitrage (prefer dividend increases, large buybacks, acquisition of franchise rights in blue-chip stocks). Pay attention to northbound capital movement and institutional seat trends on the trading leaderboard for resonance effects."
    ]
    ```

## Usage

Run ContestTrade using the Command Line Interface (CLI):

```bash
python -m cli.main run
```

The CLI allows interactive market selection and provides results summaries, detailed research reports, and data analysis reports in the `contest_trade/agents_workspace/results` directory.

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

## 🌟 Vision & Roadmap

We aim to advance quantitative trading in the AGI era, leveraging the open-source community to create a stable, trustworthy, and scalable agent trading framework.

### Roadmap

**V1.1 (Finished):**
*   Data Source Module Decoupling
*   Optimized CLI Logging and Experience

**V2.0 (Finished):**
*   US Stock Market Access
*   Expanded Factors and Signal Sources

**Future Plans:**
*   Support for Hong Kong and other Markets
*   Visual Backtesting and Analysis Interface
*   Scale up More Agents

## Contributing

Join our open-source community! Contributions are welcome in all forms.

*   **Developers:** Refer to the [Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md).
*   **Non-Code:**
    *   Suggest features or report bugs on the [Issues page](https://github.com/FinStep-AI/ContestTrade/issues).
    *   Provide testing feedback.

## Star History

<div align="center">
  <a href="https://star-history.com/#FinStep-AI/ContestTrade&Date">
    <img src="https://api.star-history.com/svg?repos=FinStep-AI/ContestTrade&type=Date" alt="Star History Chart" style="width: 80%;">
  </a>
</div>

## Disclaimer

**Important:** ContestTrade is for educational and research purposes only and does not provide investment advice.

**Risk Warning:**

*   **Market Risk:** The framework's outputs are not financial advice. Trading signals are based on historical data and AI models.
*   **Data Accuracy:** Data sources may have delays or inaccuracies; reliability is not guaranteed.
*   **Model Hallucination:** AI models have limitations; we do not guarantee the information generated.
*   **Liability:** The developers are not liable for losses from using this framework.
    Invest with caution.

**Before trading, understand the risks.**

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