# ContestTrade: Revolutionizing AI-Driven Trading

ContestTrade is a cutting-edge, multi-agent trading system designed to autonomously identify and capitalize on event-driven investment opportunities.  For more details, visit the original repository: [https://github.com/FinStep-AI/ContestTrade](https://github.com/FinStep-AI/ContestTrade).

[![arXiv](https://img.shields.io/badge/arXiv-2508.00554-B31B1B?logo=arxiv)](https://arxiv.org/abs/2508.00554)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)](https://www.python.org/downloads/release/python-3100/)
[![Reddit](https://img.shields.io/badge/Reddit-ContestTradeCommunity-orange?logo=reddit&logoColor=white)](https://www.reddit.com/r/ContestTradeCommunity/?feed=home)
[![WeChat](https://img.shields.io/badge/WeChat-ContestTrade-brightgreen?logo=wechat&logoColor=white)](./assets/wechat.png)

[English](README.md) | [中文](README_cn.md)

---

## Key Features

*   **Autonomous Stock Selection:** Automatically scans the entire market to generate tradable stock lists, eliminating the need for manual screening.
*   **Event-Driven Strategy:** Focuses on investment opportunities triggered by catalysts such as news, announcements, capital flows, and policy changes.
*   **Customizable Configuration:** Supports user-defined agent research preferences and strategies to adapt to diverse investment styles.
*   **Multi-Agent Framework:** Employs a dual-contest framework to refine decisions based on robust insights.
*   **Multi-Market Support:** Including support for US stock market data.

## Framework Overview

ContestTrade operates through a two-stage pipeline, mirroring the decision-making process of an investment firm.

1.  **Data Processing Stage:** The **Data Team** processes raw market data from multiple sources. Multiple Data Analysis Agents work in parallel to refine this raw data into structured "textual factors." An internal contest mechanism then constructs an optimal "factor portfolio."

2.  **Research and Decision-Making Stage:** The optimal factor portfolio is passed to the **Research Team**. Multiple Research Agents, each with unique "Trading Beliefs," conduct parallel analyses and submit trading proposals. A second internal contest synthesizes a reliable asset allocation strategy.

<p align="center">
  <img src="assets/architecture.jpg" style="width: 90%; height: auto;">
</p>

## Installation

Follow these steps to install and run ContestTrade:

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
docker run -it --rm --name contest_trade -v $(pwd)/config.yaml:/ContestTrade/config.yaml finstep/contesttrade:v1.1
```

## Configuration

Configure the necessary API keys and LLM parameters in `config_us.yaml`.

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

> **Note:** You need to apply for the LLM and VLM API.

## Preference - Trading Belief Configuration

Customize agent behavior via the `contest_trade/config/belief_list.json` file. This allows you to tailor the system to your specific investment style.

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

Run ContestTrade via the Command Line Interface (CLI):

```bash
python -m cli.main run
```

Interactive terminal interface allows market selection and analysis time.  View results summaries and detailed reports. All reports will be saved in the `contest_trade/agents_workspace/results` directory in Markdown format.

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

The project aims to leverage the power of the open-source community and explore quantitative trading in the AGI era.

### Roadmap

**V1.1 (Finished): Framework Stability Enhancement & Core Experience Optimization**
- [✓] Decoupled data source module (`data-provider` refactor).
- [✓] Optimized CLI logging and interaction experience.

**V2.0 (Finished): Market and Function Expansion**
- [✓] Support for **US stock** market data.
- [✓] Introduced richer factors and signal sources.

**Future Plans:**
- [ ] Support for Hong Kong stocks and other markets
- [ ] Visual backtesting and analysis interface
- [ ] Support for scaling up more agents

## Contributing

Contribute to ContestTrade!  Refer to the **[Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md)** for developer guidelines.

We value both code and non-code contributions:

*   Propose feature suggestions or report bugs via the [Issues page](https://github.com/FinStep-AI/ContestTrade/issues)
*   Provide feedback on testing results and user experience.

## Star History

<div align="center">
  <a href="https://star-history.com/#FinStep-AI/ContestTrade&Date">
    <img src="https://api.star-history.com/svg?repos=FinStep-AI/ContestTrade&type=Date" alt="Star History Chart" style="width: 80%;">
  </a>
</div>

## Disclaimer

**Important Disclaimer:** This project is for academic and educational purposes only and does not provide investment advice.

**Risk Warning:**

*   **Market Risk:** The project's outputs are based on historical data and AI model deductions and should not be considered investment advice.
*   **Data Accuracy:** Data sources may have inaccuracies.
*   **Model Hallucination:** AI models have limitations.
*   **Liability:** Developers are not liable for losses.

**Always understand the associated risks before using this framework for trading decisions.**

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