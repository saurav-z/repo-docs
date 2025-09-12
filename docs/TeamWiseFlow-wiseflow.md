# Wiseflow: Your AI Chief Intelligence Officer 🚀

**Tired of information overload? Wiseflow is your AI-powered solution, extracting the valuable insights you need from a sea of data.** Learn more about Wiseflow on its [original GitHub repository](https://github.com/TeamWiseFlow/wiseflow).

**[English](README_EN.md) | [日本語](README_JP.md) | [한국어](README_KR.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [العربية](README_AR.md)**

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TeamWiseFlow/wiseflow)

## Key Features of Wiseflow:

*   **Wide Search Focus:** Unlike "deep search" tools, Wiseflow excels at broad information gathering for industry analysis, background checks, and lead generation.
*   **Customizable Search Sources:** Configure specific search sources for your focus points, including Bing, GitHub, arXiv, and eBay.
*   **AI-Driven Perspective:**  Set roles and objectives to guide the LLM in analyzing information from a specific viewpoint.
*   **Custom Extraction Templates:** Create forms within the Pocketbase interface to define specific data extraction fields.
*   **Social Media Insights:** Find relevant content and creator information on social platforms.

## 💰  Discount on OpenAI Models

Get a 10% discount on OpenAI models through the Wiseflow application (via AiHubMix).

**Note:** To access this discount, switch to the aihubmix branch (see the [README](https://github.com/TeamWiseFlow/wiseflow/blob/aihubmix/README.md) for details).

## 🔥  Wiseflow 4.1 - What's New?

###  🔍 Custom Search Sources
Wiseflow 4.1 now supports custom search sources, including Bing, GitHub, Arxiv, and eBay. Native APIs are used for direct integration without needing third-party services.

<img src="docs/select_search_source.gif" alt="search_source" width="360">

### 🧠 AI Perspective
Configure roles and objectives for your focus point to guide the LLM during analysis.

### ⚙️ Custom Extraction Mode
Create your own form templates to extract data based on your defined fields.

### 👥 Social Media
Find relevant information and creators on social platforms.

<img src="docs/find_person_by_wiseflow.png" alt="find_person_by_wiseflow" width="720">

**For more details on version 4.1, see the [CHANGELOG](CHANGELOG.md).**

## ✋  What Sets Wiseflow Apart?

*   **Comprehensive Data Sources:** Access information from websites, social media (Weibo, Kuaishou), RSS feeds, and more.
*   **Intelligent HTML Processing:** Automatically extract and identify valuable links.
*   **User-Friendly:** "Out-of-the-box" functionality for ease of use, no XPath knowledge is required.
*   **Ongoing Development:** Benefit from high stability, efficiency, and continued improvements.
*   **More Than Just a Crawler:** Wiseflow evolves to become a comprehensive information extraction tool.

<img src="docs/wiseflow4.xscope.png" alt="4.x full scope" width="720">

(4.x architecture overview. Community contributions are welcome!)

## 🌟  Getting Started Quickly

**Follow these three simple steps to get started:**

**Windows users, install Git Bash before proceeding, [Git Bash Download](https://git-scm.com/downloads/win)**

### 📋  Download and Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/TeamWiseFlow/wiseflow.git
```

### 📥 Configure .env File

Create a `.env` file in the project root (wiseflow folder) using `env_sample` as a guide.  You only need four parameters at minimum:

```
LLM_API_KEY=""
LLM_API_BASE="https://api.siliconflow.cn/v1"
PRIMARY_MODEL=Qwen/Qwen3-14B
VL_MODEL=Pro/Qwen/Qwen2.5-VL-7B-Instruct
```

### 🚀 Run Wiseflow

```bash
cd wiseflow
uv venv # only need to run this command for the first time
source .venv/bin/activate  # Linux/macOS
# OR on Windows:
# .venv\Scripts\activate
uv sync # only need to run this command for the first time
python -m playwright install --with-deps chromium # only need to run this command for the first time
chmod +x run.sh # only need to run this command for the first time
./run.sh
```

Consult [docs/manual/manual.md](./docs/manual/manual.md) for detailed usage instructions.

## 📚  Using Wiseflow Data in Your Applications

Wiseflow stores scraped data in Pocketbase in real-time, allowing you to access and utilize the data directly through Pocketbase.

SDKs are available for Go, Javascript, Python.

Share your applications and case studies at:

- https://github.com/TeamWiseFlow/wiseflow_plus

## 🛡️  License

This project is licensed under the [Apache2.0](LICENSE) license.

For commercial collaboration, please contact: **Email：zm.zhao@foxmail.com**

- Commercial users must register with us.  The open-source version is free.

## 📬  Contact

For questions and suggestions, please open an [issue](https://github.com/TeamWiseFlow/wiseflow/issues).

## 🤝  Dependencies

Wiseflow builds upon the following open-source projects:

*   Crawl4ai (Open-source LLM Friendly Web Crawler & Scraper) https://github.com/unclecode/crawl4ai
*   MediaCrawler (xhs/dy/wb/ks/bilibili/zhihu crawler) https://github.com/NanmiCoder/MediaCrawler
*   NoDriver (Providing a blazing fast framework for web automation, webscraping, bots and any other creative ideas...) https://github.com/ultrafunkamsterdam/nodriver
*   Pocketbase (Open Source realtime backend in 1 file) https://github.com/pocketbase/pocketbase
*   Feedparser (Parse feeds in Python) https://github.com/kurtmckee/feedparser
*   SearXNG (a free internet metasearch engine which aggregates results from various search services and databases) https://github.com/searxng/searxng

## Citation

If you reference or use Wiseflow in your work, please cite it as follows:

```
Author：Wiseflow Team
https://github.com/TeamWiseFlow/wiseflow
Licensed under Apache2.0
```

## 友情链接

[<img src="docs/logos/SiliconFlow.png" alt="siliconflow" width="360">](https://siliconflow.com/)