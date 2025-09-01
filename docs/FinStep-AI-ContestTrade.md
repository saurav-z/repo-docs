<p align="center">
  <img src="assets/logo.jpg" style="width: 100%; height: auto;">
</p>
<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/abs/2508.00554" target="_blank"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2508.00554-B31B1B?logo=arxiv"/></a>
  <a href="https://opensource.org/licenses/Apache-2.0" target="_blank"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"/></a>
  <a href="https://www.python.org/downloads/release/python-3100/" target="_blank"><img alt="Python Version" src="https://img.shields.io/badge/Python-3.10+-brightgreen.svg"/></a>
  <a href="./assets/wechat.png" target="_blank"><img alt="WeChat" src="https://img.shields.io/badge/WeChat-ContestTrade-brightgreen?logo=wechat&logoColor=white"/></a>
</div>
</div>
<div align="center">
  <a href="README.md">中文</a> | <a href="README_en.md">English</a>
</div>

---

# ContestTrade: Revolutionizing AI-Driven Trading with a Multi-Agent Framework

**[ContestTrade](https://github.com/FinStep-AI/ContestTrade)** empowers you to build your own AI trading team, leveraging a cutting-edge multi-agent framework to discover and capitalize on event-driven investment opportunities.

## Key Features

*   🚀 **Automated Stock Selection:** Automatically identifies and presents a list of tradeable stocks, eliminating the need for manual screening.
*   📰 **Event-Driven Strategy:** Focuses on impactful events like news, announcements, fund flows, and policy changes to uncover significant investment opportunities.
*   🧠 **Customizable Agents:** Allows users to tailor agent research preferences and trading strategies, enabling adaptation to diverse investment approaches.
*   📊 **Structured Workflow:**  A dual-stage "contest" framework emulates the decision-making process of investment firms, ensuring robust and reliable insights.
*   💻 **Easy Deployment:**  Simple installation and configuration with clear CLI usage for seamless operation.

## Framework Overview

ContestTrade operates through a structured, dual-stage pipeline that mimics the dynamic decision-making process of an investment firm. This framework ensures that final decisions are driven by the most robust and effective insights, providing resilience and adaptability in complex markets.

<p align="center">
  <img src="assets/architecture.jpg" style="width: 90%; height: auto;">
</p>

1.  **Data Processing Stage:** Raw market data from various sources is ingested by the **Data Team**. Data Analysis Agents work in parallel to refine this data into structured "text factors." An internal contest mechanism evaluates each agent's factors, constructing an optimal "factor portfolio."

2.  **Research & Decision Stage:** The optimal factor portfolio is then passed to the **Research Team**. Research Agents, each with unique "Trading Beliefs" and financial tools, conduct in-depth analyses in parallel, submitting trading proposals. A second internal contest evaluates these proposals, ultimately producing a unified and reliable asset allocation strategy as the final output.

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/FinStep-AI/ContestTrade.git
cd ContestTrade

# 2. (Recommended) Create and activate a virtual environment
conda create -n contesttrade python=3.10
conda activate contesttrade

# 3. Install project dependencies
pip install -r requirements.txt
```

Or launch with [Docker](https://docs.n8n.io/hosting/installation/docker/) after modifying the configuration:

```
docker run -it --rm --name contest_trade -v $(pwd)/config.yaml:/ContestTrade/config.yaml finstep/contesttrade:v1.1
```

## Configuration

Configure your API keys and LLM settings in `config.yaml` before running ContestTrade.

| Configuration Item (Key) | Description                                  | Required |
| :------------------------ | :------------------------------------------- | :------: |
| `TUSHARE_KEY`             | Tushare data interface key                  |    ❌    |
| `BOCHA_KEY`               | Bocha search engine key                    |    ❌    |
| `SERP_KEY`                | SerpAPI search engine key                   |    ❌    |
| `LLM`                     | LLM API for general tasks                   |    ✅    |
| `LLM_THINKING`            | LLM API for complex reasoning                |    ❌    |
| `VLM`                     | VLM API for visual analysis                  |    ❌    |

> Note: You must obtain LLM and VLM API keys yourself and configure the URLs, API keys, and model names accordingly. The default configuration supports the AKShare data interface. You can also configure the Tushare interface and associated settings for enhanced performance.

## Preference (Stock Selection Preferences)

Each Research Agent is associated with a "trading belief." The system generates investment signals based on these beliefs, using data and tools (up to 5 signals per belief). Configuration is located in `contest_trade/config/belief_list.json`, formatted as a JSON string array.

Example 1 — Short-term event-driven (aggressive):
```json
[
  "Focus on short-term event-driven opportunities: prioritize company announcements, mergers and acquisitions, order surges, and technological breakthroughs; favor small-cap, high-volatility thematic stocks suitable for aggressive arbitrage strategies."
]
```

Example 2 — Stable events (conservative):
```json
[
  "Focus on stable, certain events: monitor dividends, repurchases, earnings forecasts, major contract signings, and policy benefits; prefer large-cap blue-chip stocks with low volatility and high certainty, suitable for stable allocation."
]
```

Default configuration:
```json
[
  "Consider the business dynamics, industry trends, and potential market impact of each company based on the provided information. Recommend a portfolio of stocks with investment potential for the next trading day for two groups: Group 1: Risk-takers (favor high-volatility, high-return, medium-to-low-cap stocks); Group 2: Conservative investors (favor low-volatility, stable-return, high-cap stocks).",
  "Based on recent events, policy adjustments, and corporate announcements, combined with market sentiment transmission paths and capital game characteristics, screen event-driven opportunities for two types of investors: Group 1: Aggressive arbitrageurs (prefer small-cap stocks with strong themes, such as expected restructuring, order surges, and technological breakthroughs); Group 2: Defensive arbitrageurs (prefer blue-chip stocks with certainty events, such as dividend increases, large-scale repurchases, and franchise acquisition). Pay attention to the resonance effects formed by Northbound Fund movements and institutional seat dynamics on the Dragon and Tiger rankings."
]
```

Instructions: Add your preferred textual descriptions, one per line, to `contest_trade/config/belief_list.json`. The system will run a corresponding Research Agent for each belief and output signals.

## Usage

Launch ContestTrade easily with the command-line interface (CLI).

```bash
python -m cli.main run
```

After starting, you will enter an interactive terminal interface to select the specific analysis time.
<p align="center">
  <img src="assets/contest_trade_cli_start.jpg" style="width: 100%; height: auto;">
</p>

After verification, you will see the following display and enter the running interface.
<p align="center">
  <img src="assets/contest_trade_cli_start_test.jpg" style="width: 100%; height: auto;">
</p>

After all Agents have finished running, you can view the signals given by the Agent in the results summary.
<p align="center">
  <img src="assets/contest_trade_cli_main.jpg" style="width: 100%; height: auto;">
</p>

You can also choose to view detailed research reports.
<p align="center">
  <img src="assets/contest_trade_cli_report.jpg" style="width: 100%; height: auto;">
</p>

<p align="center">
  <img src="assets/contest_trade_cli_research_report.jpg" style="width: 100%; height: auto;">
</p>

Or you can choose to view detailed data analysis reports.
<p align="center">
  <img src="assets/contest_trade_cli_data_report.jpg" style="width: 100%; height: auto;">
</p>

> These reports are saved in Markdown format in the `contest_trade/agents_workspace/results` directory for easy review and sharing.

## 🌟 Vision & Roadmap

We envision the imminent arrival of the AGI era and aim to leverage the power of the open-source community to explore new paradigms in quantitative trading.

This project is committed to developing more comprehensive infrastructure and richer Agents to explore the boundaries of AI's capabilities in financial trading and build a stable, reliable, and scalable Agent trading framework.

### Roadmap

**V1.1 (Finished): Framework Stability Enhancement & Core Experience Optimization**
- [✓] Core data source module decoupled, implementing multi-data source adaptor (`data-provider` refactor)
- [✓] Optimized CLI logging and interaction experience

**V2.0 (In Development): Market and Feature Expansion**
- [ ] Access **US stock** market data
- [ ] Introduce richer factor and signal sources

**Future Plans:**
- [ ] Support for Hong Kong and other markets
- [ ] Visual backtesting and analysis interface
- [ ] Support for scaling up more Agents

## Contributing

ContestTrade is a community-driven open-source project. We welcome all contributions!

Developers can refer to our **[Contributing Guidelines (CONTRIBUTING.md)](CONTRIBUTING.md)**.

We also value non-code contributions, including:
*   **Suggesting features or reporting bugs:** [Visit the Issues page](https://github.com/FinStep-AI/ContestTrade/issues)
*   **Providing feedback on your testing results:** Including test results, usage experiences, etc.

## Star History

<div align="center">
  <a href="https://star-history.com/#FinStep-AI/ContestTrade&Date">
    <img src="https://api.star-history.com/svg?repos=FinStep-AI/ContestTrade&type=Date" alt="Star History Chart" style="width: 80%;">
  </a>
</div>

## Risk Disclosure

**Important Notice:** This project `ContestTrade` is an open-source quantitative trading Agent framework research project, intended solely for academic research and educational purposes. The examples, data, and analysis results contained within do not constitute any form of investment advice.

**Risk Disclosure:**
*   **Market Risk:** This project does not constitute any form of investment, financial, legal, or tax advice. All outputs, including trading signals and analyses, are based on the AI model's projections derived from historical data and should not be used as a basis for any buying or selling decisions.
*   **Data Accuracy:** The data sources used by the framework may be subject to delays, inaccuracies, or incompleteness. We make no guarantees regarding the reliability of the data.
*   **Model Hallucinations:** AI models (including large language models) have inherent limitations and the risk of "hallucinations." We do not guarantee the accuracy, completeness, or timeliness of the information generated by the framework.
*   **Disclaimer of Liability:** The developers are not liable for any direct or indirect losses resulting from the use or inability to use this framework. Investing involves risks, and caution is advised.

**Before using this framework for any actual trading decisions, please fully understand the relevant risks.**

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