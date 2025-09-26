# Wiseflow: Your AI-Powered Information Navigator

**[English](README_EN.md) | [日本語](README_JP.md) | [한국어](README_KR.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [العربية](README_AR.md)**

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TeamWiseFlow/wiseflow)

**Wiseflow leverages Large Language Models (LLMs) to filter through vast amounts of information, delivering you the key insights you need, daily.**

[View the original repository](https://github.com/TeamWiseFlow/wiseflow)

## Key Features

*   **AI-Powered Information Extraction:**  Automatically sifts through massive datasets to surface relevant information.
*   **Customizable Search Sources:** Supports bing, github, and arxiv, allowing you to focus your searches.
*   **Role-Based Analysis:** Instructs LLMs to analyze information from specific perspectives and goals.
*   **Customizable Extraction Patterns:** Create and use custom forms to extract data with precision.
*   **Social Media Source Integration:** Find content and creators on social platforms.
*   **Chrome Browser Integration (v4.2+):** Uses your local Chrome for enhanced web scraping capabilities.
*   **Flexible LLM Compatibility:** Works with various LLM providers that support the OpenAI API format.

## What's New in v4.2

Wiseflow 4.2 enhances web scraping capabilities, by utilizing your local Chrome browser to reduce the chance of being blocked and allow for persistent user data and script execution.

*   **Enhanced Web Scraping:** Local Chrome browser integration for more robust data acquisition.
*   **Refactored Search Engine and Proxy Solutions.**
*   **Simplified Deployment:** No longer requires `playwright` installation.

## LLM Recommendations

Wiseflow recommends the following LLMs based on tests for information extraction tasks:

*   **Performance Focused:** ByteDance-Seed/Seed-OSS-36B-Instruct
*   **Cost-Conscious:** Qwen/Qwen3-14B

For more details, see the [LLM USE TEST](./test/reports/README.md).

## "Wide Search" vs. "Deep Search"

Wiseflow is designed for "wide search," focusing on broad information gathering rather than deep, targeted exploration, making it ideal for industry intelligence, background checks, and lead generation.

## Why Choose Wiseflow?

*   **Comprehensive Data Sources:** Web, social media (Weibo, Kuaishou), RSS feeds, Bing, GitHub, Arxiv, etc.
*   **Intelligent HTML Processing:** LLM-driven information extraction with a focus on relevant links.
*   **Crawler Integration:** LLMs engage during crawling to grab only relevant data, reducing the chance of being blocked.
*   **User-Friendly:** Designed for ease of use without the need for manual XPaths.
*   **Continuous Development:** Highly stable and efficient due to ongoing updates.
*   **Beyond Web Scraping:** Focused on providing useful insights.

## Getting Started

**Follow these three steps to get up and running:**

**Important:**  Starting with v4.2, you must install the Google Chrome browser (using the default installation path).

**Windows users please install git bash, follow the link below to install [git bash](https://git-scm.com/downloads/win)**

### 1. Clone the Repository and Install Dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/TeamWiseFlow/wiseflow.git
```

Next, download the appropriate PocketBase program from [PocketBase docs](https://pocketbase.io/docs/) and place it in the [.pb](./pb/) directory.

Alternatively, try using install_pocketbase.sh (for MacOS/Linux) or install_pocketbase.ps1 (for Windows).

### 2. Configure the .env File

Create a `.env` file in the project root (based on `env_sample`) and fill in the necessary settings.

For v4.x, you'll need:

*   LLM_API_KEY=""
*   LLM_API_BASE="https://api.siliconflow.cn/v1"
*   PRIMARY_MODEL=ByteDance-Seed/Seed-OSS-36B-Instruct
*   VL_MODEL=Pro/Qwen/Qwen2.5-VL-7B-Instruct

### 3. Run Wiseflow

```bash
cd wiseflow
uv venv # First time only
source .venv/bin/activate  # Linux/macOS
# or on Windows:
# .venv\Scripts\activate
uv sync # First time only
chmod +x run.sh # First time only
./run.sh
```

For detailed usage instructions, refer to [docs/manual/manual.md](./docs/manual/manual.md).

## Integrating Wiseflow Data

Wiseflow saves all scraped data in PocketBase, which can be directly accessed through the PocketBase database.

SDKs in Go, Javascript, and Python are available.

Share and promote your applications at:
* https://github.com/TeamWiseFlow/wiseflow_plus

## License

The open-source license has been updated, please see: [LICENSE](LICENSE)

For commercial partnerships, please contact **Email：zm.zhao@foxmail.com**

## Contact

For any questions or suggestions, please submit an [issue](https://github.com/TeamWiseFlow/wiseflow/issues).

## Acknowledgements

This project is built upon the following open-source projects:

*   Crawl4ai (Open-source LLM Friendly Web Crawler & Scraper) https://github.com/unclecode/crawl4ai
*   Patchright(Undetected Python version of the Playwright testing and automation library) https://github.com/Kaliiiiiiiiii-Vinyzu/patchright-python
*   MediaCrawler (xhs/dy/wb/ks/bilibili/zhihu crawler) https://github.com/NanmiCoder/MediaCrawler
*   NoDriver (Providing a blazing fast framework for web automation, webscraping, bots and any other creative ideas...) https://github.com/ultrafunkamsterdam/nodriver
*   Pocketbase (Open Source realtime backend in 1 file) https://github.com/pocketbase/pocketbase
*   Feedparser (Parse feeds in Python) https://github.com/kurtmckee/feedparser
*   SearXNG (a free internet metasearch engine which aggregates results from various search services and databases) https://github.com/searxng/searxng

## Citation

If you use or reference this project, please cite it as:

```
Author：Wiseflow Team
https://github.com/TeamWiseFlow/wiseflow
```

## Recommended Links

[<img src="docs/logos/SiliconFlow.png" alt="siliconflow" width="360">](https://siliconflow.com/)