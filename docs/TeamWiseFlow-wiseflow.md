# Wiseflow: Your AI-Powered Chief Intelligence Officer

**Uncover valuable insights from the vast ocean of information with Wiseflow, the AI-powered tool designed to filter noise and deliver the information that truly matters to you.  Discover the power of AI to extract actionable insights from web pages, social media, and more!**

[Visit the original repository:  https://github.com/TeamWiseFlow/wiseflow](https://github.com/TeamWiseFlow/wiseflow)

**[English](README_EN.md) | [日本語](README_JP.md) | [한국어](README_KR.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [العربية](README_AR.md)**

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TeamWiseFlow/wiseflow)

## Key Features

*   **Comprehensive Information Gathering:**  Access data from a wide variety of sources, including web pages, social media (Weibo and Kuaishou support), RSS feeds, Bing, GitHub, arXiv, and eBay.
*   **Intelligent Information Extraction:** Automatically extract relevant information based on your focus points, identify valuable links, and work with just a 14B parameter model.
*   **User-Friendly Interface:** Designed for ease of use, Wiseflow requires no manual Xpath configuration, making it ready to use out of the box.
*   **Customizable Search Sources:** Tailor your information gathering with support for Bing, GitHub, arXiv, and eBay search sources.
*   **AI-Driven Perspective:** Guide the AI with roles and objectives to analyze information from a specific viewpoint.
*   **Custom Extraction Templates:** Create and use custom forms to extract precisely the information you need from various sources.
*   **Social Media Creator Discovery:** Identify and find creators' profiles and content based on your focus points.
*   **PocketBase Integration:** Data is instantly stored in PocketBase, enabling easy access and integration with other applications.

## New in Version 4.1

Wiseflow 4.1 introduces exciting new features to enhance your information discovery:

*   **Custom Search Sources:** Configure precise search sources for your specific interests.  Currently supports Bing, GitHub, arXiv, and eBay.
*   **AI-Driven Analysis with Roles & Objectives:** Direct the LLM to analyze information from a specific perspective using roles and objectives.
*   **Custom Extraction Modes:** Create custom forms within the PocketBase interface to extract information with precision.
*   **Social Media Creator Discovery:** Find related content and identify content creators on social media platforms.

## "Wide Search" vs. "Deep Search"

Wiseflow is designed for **"wide search"**, focusing on broad information gathering across multiple sources, unlike the "deep search" approach, which is suitable for specific, in-depth investigations.  Wiseflow excels in scenarios like industry intelligence, background checks, and customer information gathering.

## Get Started in 3 Simple Steps!

**Windows users, please download the git bash tool beforehand and run the commands in the bash [bash download](https://git-scm.com/downloads/win)**

### 1. Download & Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/TeamWiseFlow/wiseflow.git
```

### 2. Configure .env

Create a `.env` file in the root directory, based on the `env_sample` file and provide your settings:

*   `LLM_API_KEY=""` # Your LLM service API key
*   `LLM_API_BASE="https://api.siliconflow.cn/v1"` # LLM service API endpoint (Recommended: [Referral Link](https://cloud.siliconflow.cn/i/WNLYbBpi))
*   `PRIMARY_MODEL=Qwen/Qwen3-14B` # Recommended LLM model
*   `VL_MODEL=Pro/Qwen/Qwen2.5-VL-7B-Instruct` # better to have

### 3. Run Wiseflow

```bash
cd wiseflow
uv venv # Only required on first run
source .venv/bin/activate  # Linux/macOS
# Or on Windows:
# .venv\Scripts\activate
uv sync # Only required on first run
python -m playwright install --with-deps chromium # Only required on first run
chmod +x run.sh # Only required on first run
./run.sh
```

For detailed usage instructions, see [docs/manual/manual.md](./docs/manual/manual.md).

## Accessing Your Data

All captured data is instantly saved in PocketBase. You can directly access and utilize the data through PocketBase SDKs available for various languages.

## Contribute & Develop

Explore and share your applications using Wiseflow by visiting:

*   https://github.com/TeamWiseFlow/wiseflow_plus

## Licensing

This project is licensed under [Apache2.0](LICENSE).

## Contact

For questions or suggestions, please submit an [issue](https://github.com/TeamWiseFlow/wiseflow/issues).

## Acknowledgements

This project is built upon the following open-source projects:

*   Crawl4ai
*   MediaCrawler
*   NoDriver
*   Pocketbase
*   Feedparser
*   SearXNG

## Citation

If you use this project, please cite it as:

```
Author：Wiseflow Team
https://github.com/TeamWiseFlow/wiseflow
Licensed under Apache2.0
```

## Useful Links

[<img src="docs/logos/SiliconFlow.png" alt="siliconflow" width="360">](https://siliconflow.com/)