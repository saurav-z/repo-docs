# Wiseflow: Your AI Chief Intelligence Officer 🚀

**Uncover the information that truly matters to you with Wiseflow, an AI-powered tool that filters noise from vast datasets and delivers valuable insights.** ([Original Repo](https://github.com/TeamWiseFlow/wiseflow))

**[English](README_EN.md) | [日本語](README_JP.md) | [한국어](README_KR.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [العربية](README_AR.md)**

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TeamWiseFlow/wiseflow)

## Key Features:

*   **AI-Powered Information Filtering:** Extracts and prioritizes essential information from massive datasets.
*   **Customizable Search Sources:** Supports Bing, GitHub, arXiv, and eBay, allowing focused information gathering.
*   **Role-Playing for AI Analysis:**  Instructs the LLM to analyze data from specific perspectives for tailored insights.
*   **Custom Extraction Modes:** Create custom forms to extract specific data points from sources.
*   **Social Media Creator Discovery:** Identifies content creators on social media platforms (Weibo, Kuaishou) related to your topics.
*   **Wide Search Focused:** Excels at broad information gathering for industry intelligence, background checks, and lead generation.

<img src="docs/wiseflow4.xscope.png" alt="4.x full scope" width="720">

## What Sets Wiseflow Apart?

*   **Comprehensive Data Sources:**  Gathers data from web pages, social media, RSS feeds, and search engines.
*   **Intelligent HTML Processing:** Extracts information and identifies relevant links, even with a 14B parameter LLM.
*   **User-Friendly Interface:**  Designed for ease of use, eliminating the need for manual Xpath configuration.
*   **Stable & Efficient Performance:**  Continuously updated for optimal performance and resource management.
*   **Beyond Web Scraping:**  Wiseflow is designed to be much more than just a web crawler.

## New in Version 4.1

*   **Custom Search Sources:** Configure precise search sources for focused data retrieval.
*   **AI Perspective:** Guide the LLM to analyze information from specific angles.
*   **Custom Extraction Templates:** Create custom forms for structured data extraction.
*   **Creator Discovery:** Search social platforms for content creators related to a focus point.

**For more details, see the [CHANGELOG](CHANGELOG.md).**

## Get Started Quickly

**Follow these steps to set up and run Wiseflow:**

### 1.  Download & Install Dependencies
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/TeamWiseFlow/wiseflow.git
```

### 2.  Configure PocketBase

Download PocketBase from [PocketBase Docs](https://pocketbase.io/docs/) and place it in the `.pb/` directory.  Alternatively use install_pocketbase.sh (MacOS/Linux) or install_pocketbase.ps1 (Windows)

### 3.  Set Up Your .env File

Create a `.env` file in the project root directory using `env_sample` as a template, and add your LLM, and model settings.
```bash
LLM_API_KEY="" 
LLM_API_BASE="https://api.siliconflow.cn/v1" 
PRIMARY_MODEL=Qwen/Qwen3-14B 
VL_MODEL=Pro/Qwen/Qwen2.5-VL-7B-Instruct 
```
### 4. Run Wiseflow
```bash
cd wiseflow
uv venv # Only required the first time
source .venv/bin/activate  # Linux/macOS
# or on Windows
# .venv\Scripts\activate
uv sync # Only required the first time
python -m playwright install --with-deps chromium # Only required the first time
chmod +x run.sh # Only required the first time
./run.sh
```

**For detailed usage instructions, see [docs/manual/manual.md](./docs/manual/manual.md)**

## Integrating with Your Applications

Wiseflow stores all scraped data in PocketBase. You can directly access this data via PocketBase SDKs in various languages.

## License

This project is licensed under the [Apache2.0](LICENSE) license.

## Contact

For questions or suggestions, please submit an [issue](https://github.com/TeamWiseFlow/wiseflow/issues).

## Project Based On

(List of related open source projects)

## Citation

```
Author：Wiseflow Team
https://github.com/TeamWiseFlow/wiseflow
Licensed under Apache2.0
```

## Support Wiseflow

[<img src="docs/logos/SiliconFlow.png" alt="siliconflow" width="360">](https://siliconflow.com/)