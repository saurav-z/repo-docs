# Wiseflow: Your AI-Powered Chief Intelligence Officer 🚀

**Uncover valuable insights from the vast ocean of information with Wiseflow, an AI-powered tool designed to filter noise and surface the information you truly need.**  [Access the original repository](https://github.com/TeamWiseFlow/wiseflow).

**Key Features:**

*   **Customizable Search Sources:**  Fine-tune your information gathering with support for Bing, GitHub, ArXiv, and eBay, utilizing native platform APIs.
*   **Role-Based Analysis:**  Instruct the LLM to analyze information from a specific viewpoint or with a defined objective for more targeted results.
*   **Custom Extraction Patterns:**  Create your own forms within the PocketBase interface to extract specific data points based on your needs.
*   **Social Media Creator Search:**  Identify content creators and find their profiles across social media platforms, helping you discover potential customers, partners, or investors.

## 💰 **Discounts on OpenAI Models!**

Get a 10% discount on OpenAI models using the Wiseflow application (via the AiHubMix interface).  See the [aihubmix branch README](https://github.com/TeamWiseFlow/wiseflow/blob/aihubmix/README.md) for details.

## ✨ **Wiseflow 4.1: Enhanced Intelligence & Control**

The latest version of Wiseflow brings exciting new capabilities:

*   **Customizable Search Sources:** (See the example below for configuration)
    
    <img src="docs/select_search_source.gif" alt="search_source" width="360">
*   **Role-Based Analysis:** Direct LLMs to analyze with specific perspectives and objectives.  See the [task1](test/reports/report_v4x_llm/task1) for evaluation examples.
*   **Custom Extraction Patterns:** Design custom forms in the PocketBase interface.
*   **Social Media Support:** Find creators and content based on your search criteria.

**Explore the full list of updates in the [CHANGELOG](CHANGELOG.md).**

## 🧐 **Wide Search vs. Deep Search**

Wiseflow is designed for **"wide search"**, focusing on broad information gathering.  It's the ideal tool for industry intelligence, background checks, and lead generation, offering a more efficient approach than resource-intensive "deep search" methods.

## ✋ **What Makes Wiseflow Different?**

*   **Comprehensive Data Sources:** Access web pages, social media (Weibo, Kuaishou), RSS feeds, and search engines (Bing, GitHub, ArXiv, eBay).
*   **Smart HTML Processing:** Automatically extracts key information and identifies valuable links, powered by a 14B parameter model.
*   **User-Friendly Design:** No need for manual XPath configuration; "out-of-the-box" functionality for easy use.
*   **Ongoing Development:** Expect continuous improvements and stability, with a focus on efficient resource management.
*   **Future-Proof:** The product will evolve beyond a simple crawler.

<img src="docs/wiseflow4.xscope.png" alt="4.x full scope" width="720">

(4.x architecture overview.  Community contributions are welcome to help complete the unfinished features!)

## 🚀 **Get Started in 3 Easy Steps!**

**Windows users, download Git Bash and run the commands below.** [Git Bash Download](https://git-scm.com/downloads/win)

### 1.  Clone the Repository and Install Dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/TeamWiseFlow/wiseflow.git
```

This will install `uv`.

### 2.  Install PocketBase

Download the appropriate PocketBase binary for your system from [pocketbase docs](https://pocketbase.io/docs/) and place it in the `.pb/` folder.

Alternatively, use `install_pocketbase.sh` (for MacOS/Linux) or `install_pocketbase.ps1` (for Windows).

### 3. Configure the .env File

Create a `.env` file in the project's root directory (based on the `env_sample` file) and fill in the following information:

*   `LLM_API_KEY=""`  (Your LLM service API key - any OpenAI-compatible provider is supported; if using Ollama locally, no setup is required)
*   `LLM_API_BASE="https://api.siliconflow.cn/v1"` (LLM service endpoint; consider using [this referral link](https://cloud.siliconflow.cn/i/WNLYbBpi) for a signup bonus)
*   `PRIMARY_MODEL=Qwen/Qwen3-14B` (Recommended - or a similar-sized model)
*   `VL_MODEL=Pro/Qwen/Qwen2.5-VL-7B-Instruct` (Recommended)

### 4. Run Wiseflow

```bash
cd wiseflow
uv venv # Run only for the first time
source .venv/bin/activate  # Linux/macOS
# Or on Windows:
# .venv\Scripts\activate
uv sync # Run only for the first time
python -m playwright install --with-deps chromium # Run only for the first time
chmod +x run.sh # Run only for the first time
./run.sh
```

For detailed usage instructions, please refer to [docs/manual/manual.md](./docs/manual/manual.md).

## 📚 **Accessing Data from Wiseflow**

All captured data is stored in PocketBase. You can directly interact with the PocketBase database using the SDKs available for Go, Javascript, and Python to retrieve your data.

Contribute your secondary development applications in this repository!

-   https://github.com/TeamWiseFlow/wiseflow_plus

## 🛡️ **License**

This project is licensed under the [Apache2.0](LICENSE) license.

For commercial collaborations, please contact **Email：zm.zhao@foxmail.com**.

*   Commercial customers must register their usage.  The open-source version is free forever.

## 📬 **Contact**

For questions and suggestions, please open an [issue](https://github.com/TeamWiseFlow/wiseflow/issues).

## 🤝 **Project Acknowledgements**

Wiseflow is built upon and inspired by these great open-source projects:

*   Crawl4ai (Open-source LLM Friendly Web Crawler & Scraper) https://github.com/unclecode/crawl4ai
*   MediaCrawler (xhs/dy/wb/ks/bilibili/zhihu crawler) https://github.com/NanmiCoder/MediaCrawler
*   NoDriver (Providing a blazing fast framework for web automation, webscraping, bots and any other creative ideas...) https://github.com/ultrafunkamsterdam/nodriver
*   Pocketbase (Open Source realtime backend in 1 file) https://github.com/pocketbase/pocketbase
*   Feedparser (Parse feeds in Python) https://github.com/kurtmckee/feedparser
*   SearXNG (a free internet metasearch engine which aggregates results from various search services and databases) https://github.com/searxng/searxng

## Citation

If you use or reference Wiseflow in your work, please use the following citation:

```
Author：Wiseflow Team
https://github.com/TeamWiseFlow/wiseflow
Licensed under Apache2.0
```

## 友情链接

[<img src="docs/logos/SiliconFlow.png" alt="siliconflow" width="360">](https://siliconflow.com/)