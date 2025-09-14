# Wiseflow: Your AI-Powered Information Chief 🚀

**[Original Repository](https://github.com/TeamWiseFlow/wiseflow)**

**Wiseflow helps you cut through the noise and discover the information that truly matters by leveraging large language models (LLMs) to mine valuable insights from vast amounts of data and diverse sources daily.**

*   **Key Features:**
    *   **AI-Driven Information Extraction:**  Leverages LLMs to filter and extract key information from massive datasets.
    *   **Customizable Search Sources:** Integrates with Bing, GitHub, Arxiv, and eBay for precise information gathering.
    *   **Role-Playing & Goal-Oriented Analysis:** Allows you to guide the LLM's analysis with specific roles and objectives.
    *   **Customizable Extraction Modes:** Create custom forms to extract data in structured formats.
    *   **Social Media Creator Discovery:** Identify and analyze content creators on social platforms.
    *   **Wide Search Approach:** Focused on broad information gathering, ideal for industry intelligence, background checks, and lead generation.
    *   **Versatile Data Sources:** Gathers data from various sources like web pages, social media (Weibo, Kuaishou), RSS feeds, and search engines.
    *   **Pocketbase Integration:** All scraped data is stored in Pocketbase for easy access and integration with your applications.

## 💰 Discounted OpenAI Model Access

Enjoy a 10% discount on all OpenAI models within the Wiseflow application through the AiHubMix service.  See the [aihubmix branch README](https://github.com/TeamWiseFlow/wiseflow/blob/aihubmix/README.md) for details.

## ✨ What's New in Wiseflow 4.1?

Wiseflow 4.1 introduces exciting new features:

*   **Custom Search Sources:** Configure search sources for focused information gathering (Bing, GitHub, Arxiv, eBay).
*   **Role-Playing & Goal-Oriented Analysis:** Direct LLMs with specific roles and objectives for better insights.
*   **Custom Extraction Modes:** Create and use custom forms for structured data extraction.
*   **Social Media Creator Discovery:**  Find creators and their information on social platforms, ideal for lead generation.

**See the [CHANGELOG](CHANGELOG.md) for detailed version updates.**

## 🧐 'Wide Search' vs. 'Deep Search'

Wiseflow focuses on "wide search," prioritizing broad information gathering over deep exploration, making it perfect for tasks like industry intelligence, competitor analysis, and lead generation.

## ✋ What Sets Wiseflow Apart?

*   **Comprehensive Data Acquisition:**  Web pages, social media (Weibo, Kuaishou), RSS feeds, and search engines are all supported.
*   **Optimized HTML Processing:**  Efficiently extracts information and identifies valuable links.
*   **User-Friendly Interface:** No need for manual Xpath configuration; "out-of-the-box" functionality.
*   **High Stability and Availability:**  Consistent performance and resource efficiency through continuous updates.
*   **Future-Proof Design:**  A project that goes beyond being just a "crawler".

## 🌟 Quick Start Guide

**Follow these three simple steps to get started!**

**Windows users:**  Download and install Git Bash before proceeding.  [Git Bash Download](https://git-scm.com/downloads/win)

### 📋 Clone the Repository & Install Dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/TeamWiseFlow/wiseflow.git
```

This will install `uv`.

Next, download the correct PocketBase executable for your system from [PocketBase Docs](https://pocketbase.io/docs/) and place it in the `.pb/` folder.
Alternatively, try running install_pocketbase.sh (for MacOS/Linux) or install_pocketbase.ps1 (for Windows) to install it.

### 📥 Configure Your Environment

Create a `.env` file in the project root directory based on the `env_sample` file and fill in your settings.  The minimum configuration requires:

*   `LLM_API_KEY=""` (Your LLM service API key - any OpenAI-compatible service works)
*   `LLM_API_BASE="https://api.siliconflow.cn/v1"` (Your LLM service API endpoint - SiliconFlow is recommended.  Use my [referral link](https://cloud.siliconflow.cn/i/WNLYbBpi) for a bonus!)
*   `PRIMARY_MODEL=Qwen/Qwen3-14B` (Recommended: Qwen3-14B or similar-sized model)
*   `VL_MODEL=Pro/Qwen/Qwen2.5-VL-7B-Instruct` (Recommended, but optional)

### 🚀 Run Wiseflow

```bash
cd wiseflow
uv venv # Run only the first time
source .venv/bin/activate  # Linux/macOS
# Or on Windows:
# .venv\Scripts\activate
uv sync # Run only the first time
python -m playwright install --with-deps chromium # Run only the first time
chmod +x run.sh # Run only the first time
./run.sh
```

For detailed instructions, see [docs/manual/manual.md](./docs/manual/manual.md).

## 📚 Integrating with Your Applications

Wiseflow saves all scraped data in Pocketbase.  Use Pocketbase SDKs (Go/Javascript/Python and more) to access and utilize the data in your own projects.

Share your applications at: [https://github.com/TeamWiseFlow/wiseflow_plus](https://github.com/TeamWiseFlow/wiseflow_plus)

## 🛡️ License

This project is licensed under the [Apache2.0](LICENSE) license.

For commercial partnerships, please contact: **Email: zm.zhao@foxmail.com**

## 📬 Contact

For questions or suggestions, please open an [issue](https://github.com/TeamWiseFlow/wiseflow/issues).

## 🤝 Acknowledgements

Wiseflow is built upon these outstanding open-source projects:

*   Crawl4ai (Open-source LLM Friendly Web Crawler & Scraper) https://github.com/unclecode/crawl4ai
*   MediaCrawler (xhs/dy/wb/ks/bilibili/zhihu crawler) https://github.com/NanmiCoder/MediaCrawler
*   NoDriver (Providing a blazing fast framework for web automation, webscraping, bots and any other creative ideas...) https://github.com/ultrafunkamsterdam/nodriver
*   Pocketbase (Open Source realtime backend in 1 file) https://github.com/pocketbase/pocketbase
*   Feedparser (Parse feeds in Python) https://github.com/kurtmckee/feedparser
*   SearXNG (a free internet metasearch engine which aggregates results from various search services and databases) https://github.com/searxng/searxng

## Citation

If you use or reference Wiseflow in your work, please cite it as follows:

```
Author：Wiseflow Team
https://github.com/TeamWiseFlow/wiseflow
Licensed under Apache2.0
```

## 友情链接
[<img src="docs/logos/SiliconFlow.png" alt="siliconflow" width="360">](https://siliconflow.com/)