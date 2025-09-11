# Wiseflow: Your AI-Powered Chief Information Officer 🚀

**Uncover the insights you need from a flood of information using Wiseflow, an AI-powered tool designed to filter the noise and deliver the most relevant information to you. [Explore Wiseflow on GitHub](https://github.com/TeamWiseFlow/wiseflow)**

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TeamWiseFlow/wiseflow)

Wiseflow helps you cut through the clutter of the internet to discover the information that matters most to you.

## Key Features

*   **Customizable Search Sources:** Tailor your information gathering with support for Bing, GitHub, arXiv, and eBay, using native platform APIs.
*   **Role-Based AI Analysis:**  Guide the AI with specific roles and objectives for more focused and insightful information extraction.
*   **Customizable Extraction Modes:** Create and configure custom forms within Pocketbase to extract specific data points based on your needs.
*   **Social Media Content Discovery:**  Identify relevant content and creators on social platforms, helping you find potential clients, partners, or investors.
*   **Wide Search Focus:** Unlike "deep search" tools, Wiseflow excels at broad information gathering across various sources, perfect for industry intelligence and background research.
*   **Multi-Platform Data Acquisition:** Access information from websites, social media (Weibo, Kuaishou), RSS feeds, and more.
*   **User-Friendly Interface:** Designed for ease of use, Wiseflow requires no manual Xpath configuration, making it accessible to everyone.
*   **Ongoing Development:** Benefit from continuous improvements and high stability.

## 💰 Discounted OpenAI Models

Get 10% off on all OpenAI models within the Wiseflow application through the AiHubMix service.  For details, see the [aihubmix branch README](https://github.com/TeamWiseFlow/wiseflow/blob/aihubmix/README.md)

## 🆕 What's New in Wiseflow 4.1?

This version introduces exciting new features:

*   **Enhanced Search:** Explore diverse search sources including Bing, Github, Arxiv, and Ebay.
*   **AI-Driven Insights:** Define roles and objectives for AI analysis, unlocking valuable perspectives.
*   **Custom Extraction:** Create your own forms for precise data extraction.
*   **Social Media Support:** Find creators using your focus points.

**For a complete list of updates, see the [CHANGELOG](CHANGELOG.md).**

## 🚀 Getting Started Quickly

**Follow these three easy steps to begin using Wiseflow:**

**Windows users should install Git Bash before proceeding. [Download Git Bash](https://git-scm.com/downloads/win)**

### 1.  Download and Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/TeamWiseFlow/wiseflow.git
```
This installs uv.

Then, download the appropriate PocketBase program from the [PocketBase Docs](https://pocketbase.io/docs/) and place it in the [.pb](./pb/) directory.

You can also use `install_pocketbase.sh` (for MacOS/Linux) or `install_pocketbase.ps1` (for Windows).

### 2.  Configure the .env File

Create a `.env` file in the Wiseflow project root directory by referring to `env_sample` and filling in your settings.

For version 4.x, you only need these parameters:

-   `LLM_API_KEY=""`
-   `LLM_API_BASE="https://api.siliconflow.cn/v1"`
-   `PRIMARY_MODEL=Qwen/Qwen3-14B`
-   `VL_MODEL=Pro/Qwen/Qwen2.5-VL-7B-Instruct`

### 3.  Run Wiseflow

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

For detailed usage instructions, see the [manual](docs/manual/manual.md).

## 📚 Using Wiseflow Data in Your Applications

All scraped data is stored in PocketBase, allowing you to directly interact with the database. PocketBase has SDKs for Go, Javascript, and Python.

Share and promote your applications at: [https://github.com/TeamWiseFlow/wiseflow_plus](https://github.com/TeamWiseFlow/wiseflow_plus)

## 🛡️ License

Wiseflow is licensed under the [Apache2.0](LICENSE).

For commercial collaboration, please contact **Email: zm.zhao@foxmail.com**

## 📬 Contact

For questions or suggestions, please submit an [issue](https://github.com/TeamWiseFlow/wiseflow/issues).

## 🤝 Open Source Projects

Wiseflow builds upon the following open-source projects:

*   Crawl4ai https://github.com/unclecode/crawl4ai
*   MediaCrawler https://github.com/NanmiCoder/MediaCrawler
*   NoDriver https://github.com/ultrafunkamsterdam/nodriver
*   Pocketbase https://github.com/pocketbase/pocketbase
*   Feedparser https://github.com/kurtmckee/feedparser
*   SearXNG https://github.com/searxng/searxng

## Citation

Please cite the following if you use Wiseflow:

```
Author：Wiseflow Team
https://github.com/TeamWiseFlow/wiseflow
Licensed under Apache2.0
```

## 友情链接

[<img src="docs/logos/SiliconFlow.png" alt="siliconflow" width="360">](https://siliconflow.com/)