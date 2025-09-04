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

# ContestTrade: Unleash AI-Powered Event-Driven Trading 

**ContestTrade** is an innovative, multi-agent trading framework that empowers you to build and deploy your own AI trading team for event-driven investment strategies.  Check out the [original repository](https://github.com/FinStep-AI/ContestTrade) for more details.

## Key Features

*   **Automated Stock Selection:**  Scans the entire market to identify and generate tradable stock lists, eliminating the need for manual screening.
*   **Event-Driven Strategy:** Focuses on opportunities triggered by catalysts like news, announcements, capital flows, and policy changes to capitalize on significant information impacts.
*   **Personalized Configuration:** Allows users to define agent research preferences and trading strategies, adapting to various investment styles.
*   **Multi-Agent Architecture:** Employs a two-stage pipeline with a Data Team and Research Team, using an internal contest mechanism to refine decisions and ensure robustness.

## Overview: How ContestTrade Works

ContestTrade uses a two-stage pipeline mirroring a professional investment firm's decision-making process.

1.  **Data Processing Stage:**  Raw market data is processed by the **Data Team**, where multiple Data Analysis Agents extract textual factors. An internal contest mechanism evaluates these factors to create an optimal "factor portfolio."

2.  **Research and Decision-Making Stage:**  The factor portfolio is analyzed by the **Research Team**. Research Agents, with unique "Trading Beliefs," analyze factors and propose trades. Another internal contest synthesizes these proposals into a unified asset allocation strategy.

## Installation

Get started with ContestTrade quickly:

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

Alternatively, deploy with Docker:

```bash
docker run -it --rm --name contest_trade -v $(pwd)/config.yaml:/ContestTrade/config.yaml finstep/contesttrade:v1.1
```

## Configuration

Configure API keys and LLM parameters in `config_us.yaml`.

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

>  You'll need to obtain your own LLM and VLM API keys.  Fill in the necessary URLs, API Keys, and model names according to the platform and model you choose.

## Customizing Trading Beliefs (Preferences)

Customize each Research Agent's "trading belief" via the `contest_trade/config/belief_list.json` file.  Here are two examples:

**Example 1: Short-Term Event-Driven (Aggressive)**

```json
[
  "Focus on short-term event-driven opportunities: prioritize company announcements, M&A and restructuring, sudden order increases, technological breakthroughs and other catalysts; prefer mid/small-cap, high-volatility thematic stocks, suitable for aggressive arbitrage strategies."
]
```

**Example 2: Stable Events (Conservative)**

```json
[
  "Focus on stable, high-certainty events: pay attention to dividends, buybacks, earnings forecast confirmations, major contract signings and policy tailwinds; prefer large-cap blue-chips with low volatility and high certainty, suitable for conservative allocation."
]
```

## Usage

Run ContestTrade from the CLI:

```bash
python -m cli.main run
```

The CLI provides an interactive interface to select markets and view analysis results, including detailed research and data reports. Reports are saved in Markdown format within the `contest_trade/agents_workspace/results` directory.

## 🌟 Vision & Roadmap

ContestTrade aims to pioneer quantitative trading in the AGI era, leveraging the power of open-source collaboration.

### Roadmap

*   **V1.1 (Finished):** Framework Stability Enhancement & Core Experience Optimization
    *   [✓] Decoupled data source module for multiple data source adaptors.
    *   [✓] Optimized CLI logging and interaction experience.
*   **V2.0 (Finished):** Market and Function Expansion
    *   [✓] Access to US stock market data.
    *   [✓] Introduced richer factors and signal sources.
*   **Future Plans:**
    *   Support for Hong Kong and other markets.
    *   Visual backtesting and analysis interface.
    *   Scale up agent capabilities.

## Contributing

Contribute to ContestTrade! We welcome contributions of all types. See the [Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md) for developer guidance.

We also welcome non-code contributions, such as:

*   Proposing feature suggestions or reporting bugs: [Go to Issues page](https://github.com/FinStep-AI/ContestTrade/issues)
*   Providing feedback on testing results.

## Star History

<div align="center">
  <a href="https://star-history.com/#FinStep-AI/ContestTrade&Date">
    <img src="https://api.star-history.com/svg?repos=FinStep-AI/ContestTrade&type=Date" alt="Star History Chart" style="width: 80%;">
  </a>
</div>

## Disclaimer

**Important:** This project is a research framework for educational purposes. No investment advice is provided.

**Risk Warning:**

*   **Market Risk:** No investment advice. Trading signals and analyses are AI-generated and based on historical data.
*   **Data Accuracy:** Data sources may be delayed, inaccurate, or incomplete.
*   **Model Hallucination:** AI models have limitations, and results may be inaccurate.
*   **Liability:** Developers are not liable for losses from using this framework. Investing involves risk.

**Before trading, fully understand the risks.**

## Citation

Cite ContestTrade if you use it in your research:

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