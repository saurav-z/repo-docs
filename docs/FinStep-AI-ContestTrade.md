<p align="center">
  <img src="assets/logo.jpg" style="width: 100%; height: auto;">
</p>

<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/abs/2508.00554" target="_blank"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2508.00554-B31B1B?logo=arxiv"/></a>
  <a href="https://opensource.org/licenses/Apache-2.0" target="_blank"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"/></a>
  <a href="https://www.python.org/downloads/release/python-3100/" target="_blank"><img alt="Python Version" src="https://img.shields.io/badge/Python-3.10+-brightgreen.svg"/></a>
  <a href="./assets/wechat.png" target="_blank"><img alt="WeChat" src="https://img.shields.io/badge/WeChat-ContestTrade-brightgreen?logo=wechat&logoColor=white"/></a>
</div>

<div align="center">
  <a href="README.md">中文</a> | <a href="README_en.md">English</a>
</div>

---

# ContestTrade: AI-Powered Multi-Agent Trading for Event-Driven Stock Selection

**ContestTrade** is an innovative multi-agent trading framework that empowers you to build your own AI trading team, automatically identifying and capitalizing on event-driven investment opportunities in the stock market. Check out the [original repository](https://github.com/FinStep-AI/ContestTrade) for more details.

## Key Features

*   **Automated Stock Screening:**  Scans the entire market to generate a list of tradeable stocks, eliminating the need for manual filtering.
*   **Event-Driven Strategy:**  Focuses on investment opportunities triggered by news, announcements, fund flows, and policy changes.
*   **Customizable Agents:** Allows users to define research preferences and strategies, adapting to various investment styles.
*   **Internal Contest Mechanism:** A dual-stage competition ensures the most robust and effective insights drive final investment decisions.

## Overview

ContestTrade operates through a structured, two-stage pipeline that mimics the decision-making process of an investment firm.  This framework leverages a "contest" mechanism to drive a data analysis team and research team, ensuring optimal trading recommendations.

**1. Data Processing Stage:** Raw market data is fed to the **Data Team**, where multiple Data Analysis Agents extract structured "text factors". An internal competition evaluates these factors, leading to an optimal "factor portfolio."

**2. Research & Decision Stage:** The optimized factor portfolio is passed to the **Research Team**. Research Agents, with their unique "Trading Beliefs" and financial tools, analyze the factors to submit trading proposals. A second internal competition assesses these proposals, resulting in a unified asset allocation strategy.

## Installation

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

Alternatively, configure and launch using [Docker](https://docs.n8n.io/hosting/installation/docker/):

```bash
docker run -it --rm --name contest_trade -v $(pwd)/config.yaml:/ContestTrade/config.yaml finstep/contesttrade:v1.1
```

## Configuration

Before running ContestTrade, configure API keys and LLM parameters in `config.yaml`:

| Key            | Description                         | Required |
| :------------- | :---------------------------------- | :------: |
| `TUSHARE_KEY`  | Tushare Data Interface Key          |    ❌     |
| `BOCHA_KEY`    | Bocha Search Engine Key            |    ❌     |
| `SERP_KEY`     | SerpAPI Search Engine Key          |    ❌     |
| `LLM`          | LLM API for general tasks           |    ✅     |
| `LLM_THINKING` | LLM API for complex reasoning      |    ❌     |
| `VLM`          | VLM API for visual analysis        |    ❌     |

>  *Note:  You must obtain your own LLM and VLM API keys.  The configuration file defaults to the AKShare data interface; you can configure Tushare for improved performance.*

## Preference (Stock Selection Preferences)

Define "Trading Beliefs" for each Research Agent in `contest_trade/config/belief_list.json`.

*   **Example 1 (Aggressive):**

    ```json
    [
      "Focus on short-term event-driven opportunities: Prioritize company announcements, M&A, order surges, and technological breakthroughs; favor small-cap, high-volatility stocks for aggressive arbitrage strategies."
    ]
    ```

*   **Example 2 (Conservative):**

    ```json
    [
      "Focus on stable, confirmed events: Consider dividends, share buybacks, earnings forecasts, major contract signings, and favorable policies; favor large-cap, low-volatility stocks for a stable portfolio."
    ]
    ```

*   **Default Configuration:**

    ```json
    [
      "Based on the provided information, comprehensively consider the business dynamics, industry trends, and potential market impact of each company. Recommend a portfolio of stocks with investment potential for the next trading day for two groups: Group 1: Risk-takers (prefer high-volatility, high-return, mid-to-low market-cap stocks); Group 2: Conservative investors (prefer low-volatility, stable-return, high-market-cap stocks).",
      "Based on recent events, policy adjustments, and corporate announcements, along with market sentiment and capital flow characteristics, select event-driven opportunities for two types of investors: Group 1: Aggressive arbitrageurs (prefer small-cap stocks with expectations of restructuring, order surges, and technological breakthroughs); Group 2: Defensive arbitrageurs (prefer blue-chip stocks with confirmed events such as dividend increases, large-scale repurchases, and the acquisition of franchises). Pay attention to the resonance effect formed by Northbound fund movements and the movements of institutional seats on the Dragon and Tiger list."
    ]
    ```

*   **Customize:** Add your preferred text descriptions to `contest_trade/config/belief_list.json` for each Research Agent to run accordingly.

## Usage

Run ContestTrade via the command-line interface (CLI):

```bash
python -m cli.main run
```

Follow the prompts to select an analysis time and view agent signals and detailed reports.
(Include images from original README, like those of the CLI screens, here for better visualization)

## Vision & Roadmap

We aim to explore new paradigms for quantitative trading in the AGI era by leveraging the open-source community.

*   **V1.1 (Finished):** Framework stability and core experience improvements.
    *   [✓] Decoupled core data source module for multi-source adaptors.
    *   [✓] Optimized CLI logging and interaction.

*   **V2.0 (In Development):** Market and Functionality Expansion.
    *   [ ] Integrate **US Stock** Market Data.
    *   [ ] Introduce Richer Factors and Signal Sources.

*   **Future Plans:**
    *   [ ] Support for Hong Kong and other markets.
    *   [ ] Visual backtesting and analysis interface.
    *   [ ] Scale up support for more agents.

## Contributing

Contribute by following our **[Contributing Guide](CONTRIBUTING.md)**. We welcome:
*   Feature suggestions and bug reports via the [Issues page](https://github.com/FinStep-AI/ContestTrade/issues).
*   Feedback on your test results and experiences.

## Star History

(Embed the Star History Chart from the original README here)

## Risk Disclaimer

**Important:** This project is for research and educational purposes. All examples and results do not constitute investment advice.

**Risk Warning:**

*   **Market Risk:**  This project is for research only and doesn't provide investment, financial, legal, or tax advice. Trading signals and analysis are based on AI model results and are not recommendations to buy or sell.
*   **Data Accuracy:** Data may be delayed, inaccurate, or incomplete. We do not guarantee data reliability.
*   **Model Limitations:**  AI models have inherent limitations and can "hallucinate." We do not guarantee the accuracy of generated information.
*   **Responsibility:** Developers are not liable for any direct or indirect losses from using this framework. Investment involves risk; invest cautiously.

**Before using this framework for actual trading decisions, understand the risks.**

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
```
Key improvements and summary of changes:

*   **SEO Optimization:**  Includes relevant keywords like "AI," "multi-agent," "trading," "event-driven," "stock selection," "quantitative trading" throughout the document.
*   **Hook:** The one-sentence hook is at the beginning of the readme, immediately grabbing attention.
*   **Clear Structure:** The use of headings, subheadings, bullet points, and concise language makes the README much more readable.
*   **Summarization:**  Condenses the original text while retaining all key information.
*   **Actionable Instructions:** The installation and usage sections are clear and easy to follow.
*   **Emphasis on Value Proposition:** Highlights the key benefits of using ContestTrade.
*   **Formatting:**  Uses Markdown formatting effectively for better readability and visual appeal.
*   **Removed Redundancy:**  Eliminated unnecessary wording and focused on the core information.
*   **Vision and Roadmap:**  Clear, concise, and organized into finished and future plans.
*   **Risk Disclaimer:** The risk disclaimer is prominent and well-written.
*   **Citation:** Includes the citation block.
*   **Image Placeholders:** Includes placeholders for the images for better visualization, as they're crucial for a good README.
*   **Concise Explanations:** Each section is explained concisely to capture the reader's attention.
*   **Improved Tone:** Uses more engaging language.
*   **Direct Link to Repo:** Includes the link to the original repo, making it very clear where to find the project.