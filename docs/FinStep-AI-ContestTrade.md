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

**ContestTrade** is an innovative, open-source framework empowering you to build and deploy an autonomous AI trading team that identifies and capitalizes on event-driven investment opportunities.  ([View on GitHub](https://github.com/FinStep-AI/ContestTrade))

## Key Features

*   **Automated Stock Selection:** Scans the entire market to generate tradable stock lists without manual intervention.
*   **Event-Driven Strategy:** Focuses on market catalysts like news, announcements, and policy changes to identify impactful opportunities.
*   **Customizable Agents:** Supports user-defined trading strategies and preferences to tailor the framework to different investment styles.
*   **Multi-Agent Architecture:**  Employs a dual-contest framework for robust decision-making, simulating an investment firm's workflow.
*   **Modular Design:** Easy to install and configure, with support for various data sources.
*   **Cross-Platform Support:** Run ContestTrade from the command line, or utilize Docker for easy deployment.
*   **Detailed Reporting:**  Generates comprehensive reports in Markdown format for analysis and sharing.

## Project Overview

ContestTrade is a multi-agent trading framework designed to automatically discover and evaluate event-driven investment opportunities, ultimately providing actionable asset allocation recommendations.  The system simulates an investment firm's workflow in two key stages:

1.  **Data Processing Stage:** The **Data Team** refines raw market data using multiple Data Analysis Agents, generating textual factors. An internal contest mechanism then evaluates these factors to create an optimal "factor portfolio."
2.  **Research and Decision-Making Stage:** The optimal factor portfolio is passed to the **Research Team**. Multiple Research Agents, each with unique "Trading Beliefs," analyze the factors and propose trading strategies. A second internal contest synthesizes these proposals for a final, reliable asset allocation strategy.

## Installation

Follow these steps to get started:

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

Alternatively, deploy using Docker:

```bash
docker run -it --rm --name contest_trade -v $(pwd)/config.yaml:/ContestTrade/config.yaml finstep/contesttrade:v1.1
```

## Configuration

Before running ContestTrade, configure your API keys and LLM parameters in `config_us.yaml`.

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

> **Note:**  Obtain your own LLM and VLM API keys and fill in the corresponding URLs and model names.

## Preferences

Customize each Research Agent's "trading belief" by editing the `contest_trade/config/belief_list.json` file.  The file contains a JSON array of strings defining the agent's investment style.

**Examples:**

*   **Short-term event-driven (Aggressive):**
    ```json
    [
      "Focus on short-term event-driven opportunities: prioritize company announcements, M&A and restructuring, sudden order increases, technological breakthroughs and other catalysts; prefer mid/small-cap, high-volatility thematic stocks, suitable for aggressive arbitrage strategies."
    ]
    ```

*   **Stable events (Conservative):**
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

## Usage

Run ContestTrade from the command line using the CLI:

```bash
python -m cli.main run
```

Follow the prompts in the interactive terminal to select your market and view analysis results.

## Screenshots

*(Include screenshots from the original README showing CLI interaction.  Captions could include: "Market Selection," "Results Summary," "Detailed Research Report," and "Data Analysis Report.")*

## 🌟 Vision & Roadmap

ContestTrade aims to push the boundaries of quantitative trading using AI, envisioning a future where AGI empowers new financial paradigms. The project is committed to developing robust infrastructure, expanding agent capabilities, and fostering a trustworthy, scalable trading framework.

### Roadmap

*   **V1.1 (Completed):** Enhanced Framework Stability & Core Experience Optimization
    *   [✓] Decoupled core data source module for multi-source adaptors.
    *   [✓] Improved CLI logging and user interaction.

*   **V2.0 (Completed):** Market and Function Expansion
    *   [✓] Added US stock market data support.
    *   [✓] Introduced richer factors and signal sources.

*   **Future Plans:**
    *   [ ] Support for Hong Kong and other markets.
    *   [ ] Implement a visual backtesting and analysis interface.
    *   [ ] Expand support for more agents.

## Contributing

ContestTrade thrives on community contributions!  We welcome:

*   **Developers:** Refer to our **[Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md)**.
*   **Non-code contributions:** Feature suggestions, bug reports, and testing feedback via the [Issues page](https://github.com/FinStep-AI/ContestTrade/issues).

## Star History

*(Keep the Star History Chart Image Here.)*

## Disclaimer

**Disclaimer:** This project is for research and educational purposes only and does not constitute investment advice.  Use at your own risk.

**Risk Warning:**

*   **Market Risk:**  Outputs are based on historical data and AI models and should not be considered investment advice.
*   **Data Accuracy:** Data sources may have delays, inaccuracies, or be incomplete.
*   **Model Hallucination:** AI models are subject to potential inaccuracies.
*   **Liability:** The developers are not liable for any losses resulting from the use of this framework.  Investing carries risk.

**Carefully consider all risks before using this framework for any trading decisions.**

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