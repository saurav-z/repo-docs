# Wiseflow: Your AI-Powered Information Navigator 🚀

**Tired of information overload? Wiseflow, your AI Chief Intelligence Officer, cuts through the noise and uncovers the valuable insights you need from a sea of data, all powered by cutting-edge AI.** [Explore the original repository on GitHub](https://github.com/TeamWiseFlow/wiseflow).

**[English](README_EN.md) | [日本語](README_JP.md) | [한국어](README_KR.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [العربية](README_AR.md)**

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TeamWiseFlow/wiseflow)

## Key Features:

*   **Wide Search Focus:** Unlike "deep search" tools, Wiseflow excels at broad information gathering for industry intelligence, background checks, and lead generation.
*   **Multi-Source Data Extraction:**  Gather insights from web pages, social media (Weibo, Kuaishou), RSS feeds, and search engines (Bing, GitHub, Arxiv, and eBay).
*   **AI-Driven Information Filtering:**  Intelligently extracts and highlights the most relevant information based on your specified focus points.
*   **Customizable Extraction:** Create custom forms within the PocketBase interface for precise information extraction based on your needs.
*   **Enhanced Search with AI:**  Leverage AI to analyze information from social media and find creator profiles.
*   **Competitive Pricing:** Enjoy 90% off OpenAI models (via AiHubMix).
*   **Open Source & Commercial Friendly:**  Use the open-source version for free, or contact the project for commercial partnerships.

## What Sets Wiseflow Apart?

*   **Comprehensive Data Sources:**  Access information across various web platforms, social media, and search engines.
*   **Intelligent Information Filtering:** Processes HTML and focuses on extracting relevant data with a 14B parameter model.
*   **User-Friendly Interface:** Designed for ease of use, eliminating the need for complex coding or manual Xpath inputs.
*   **Continuous Improvement:** Stay up-to-date with new features, enhanced stability, and optimized efficiency through regular updates.
*   **More Than Just a Crawler:** Expanding capabilities beyond simple web scraping.

## Wiseflow 4.1: What's New?

*   **Custom Search Sources:** Configure specific search engines (Bing, GitHub, Arxiv, eBay) for targeted data acquisition.
    <img src="docs/select_search_source.gif" alt="search_source" width="360">
*   **AI-Driven Perspective:**  Set roles and objectives for AI analysis to gain specific insights.  (See the [task1](test/reports/report_v4x_llm/task1) for case studies)
*   **Enhanced Social Media Search:** Easily discover content creators related to your search query.
    <img src="docs/find_person_by_wiseflow.png" alt="find_person_by_wiseflow" width="720">

**For the latest updates, see the [CHANGELOG](CHANGELOG.md).**

## Quick Start: Get Up and Running in 3 Steps

**Windows users, please download the git bash tool beforehand. [Git Bash Download](https://git-scm.com/downloads/win)**

### 1. Download and Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/TeamWiseFlow/wiseflow.git
```

### 2. Configure Your Environment

*   Create a `.env` file in the root directory, using `env_sample` as a template.
*   Set the following parameters:

    *   `LLM_API_KEY=""`  (Your LLM service API key)
    *   `LLM_API_BASE="https://api.siliconflow.cn/v1"` (Your LLM service API endpoint - consider SiliconFlow, [referral link](https://cloud.siliconflow.cn/i/WNLYbBpi) to receive a bonus)
    *   `PRIMARY_MODEL=Qwen/Qwen3-14B` (Recommended model)
    *   `VL_MODEL=Pro/Qwen/Qwen2.5-VL-7B-Instruct` (Optional Visual Language Model, highly recommended)

### 3. Launch Wiseflow

```bash
cd wiseflow
uv venv # only execute the first time
source .venv/bin/activate  # Linux/macOS
# Or on Windows:
# .venv\Scripts\activate
uv sync # only execute the first time
python -m playwright install --with-deps chromium # only execute the first time
chmod +x run.sh # only execute the first time
./run.sh
```

For detailed usage instructions, refer to [docs/manual/manual.md](./docs/manual/manual.md)

## Integrate Your Data:

Wiseflow stores data in PocketBase. Use PocketBase's Go/Javascript/Python SDKs to access and integrate the scraped data into your applications.

Explore and share your secondary development cases at: [wiseflow_plus](https://github.com/TeamWiseFlow/wiseflow_plus)

## License

Wiseflow is released under the [Apache2.0](LICENSE) license.

## Contact

For questions or suggestions, please open an [issue](https://github.com/TeamWiseFlow/wiseflow/issues).

## Acknowledgements

Wiseflow is built upon these excellent open-source projects:

*   Crawl4ai (Open-source LLM Friendly Web Crawler & Scraper) https://github.com/unclecode/crawl4ai
*   MediaCrawler (xhs/dy/wb/ks/bilibili/zhihu crawler) https://github.com/NanmiCoder/MediaCrawler
*   NoDriver (Providing a blazing fast framework for web automation, webscraping, bots and any other creative ideas...) https://github.com/ultrafunkamsterdam/nodriver
*   Pocketbase (Open Source realtime backend in 1 file) https://github.com/pocketbase/pocketbase
*   Feedparser (Parse feeds in Python) https://github.com/kurtmckee/feedparser
*   SearXNG (a free internet metasearch engine which aggregates results from various search services and databases) https://github.com/searxng/searxng

## Citation

If you use Wiseflow in your work, please cite it as:

```
Author：Wiseflow Team
https://github.com/TeamWiseFlow/wiseflow
Licensed under Apache2.0
```

## Resources

[<img src="docs/logos/SiliconFlow.png" alt="siliconflow" width="360">](https://siliconflow.com/)