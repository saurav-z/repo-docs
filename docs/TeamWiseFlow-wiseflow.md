# Wiseflow: Your AI-Powered Chief Information Officer 

**[English](README_EN.md) | [日本語](README_JP.md) | [한국어](README_KR.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [العربية](README_AR.md)**

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TeamWiseFlow/wiseflow)

**Wiseflow is an AI-powered tool that sifts through vast amounts of information to deliver the key insights you need daily.**

Tired of information overload? Wiseflow filters the noise, bringing valuable information to the surface, empowering you with the knowledge you need.

[Original Repository](https://github.com/TeamWiseFlow/wiseflow)

<img src="https://github.com/user-attachments/assets/48998353-6c6c-4f8f-acae-dc5c45e2e0e6" alt="Wiseflow Demo" width="100%">

## Key Features

*   **Web Scraping with Local Chrome:** Leverage your local Chrome browser for reliable web content extraction.
*   **Customizable Search Sources:** Integrate with Bing, GitHub, and Arxiv for targeted information gathering.
*   **AI-Driven Analysis:** Guide LLMs to analyze information from specific perspectives for more relevant results.
*   **Custom Extraction Templates:** Create custom forms within the PB interface to extract data in a structured manner.
*   **Social Media Content Discovery:** Identify content and creator information on social platforms to find potential leads.
*   **LLM Recommendation:** Provides recommendations for optimal LLM model selection based on performance and cost.
*   **Wide Search Focus:** Unlike deep search tools, Wiseflow excels at broad information gathering for industry trends, background checks, and more.
*   **PocketBase Integration:** All scraped data is stored in PocketBase for easy access and integration.

## What's New in Version 4.2 🔥

Version 4.2 enhances web scraping capabilities by directly utilizing your local Chrome browser, significantly reducing the likelihood of being blocked and improving data extraction. This version also introduces:

*   **Chrome Browser Integration:** Directly uses your local Chrome for advanced web scraping, maintaining user data and supporting page scripts.
*   **Refactored Search Engine:** Enhanced search engine capabilities.
*   **Comprehensive Proxy Solutions:** Improved proxy support for reliable data retrieval.

See the [CHANGELOG](CHANGELOG.md) for detailed updates.

## Getting Started

**Follow these steps to start using Wiseflow:**

**Important:** From version 4.2, ensure you have Google Chrome installed using the default installation path.

**Windows users:** Download Git Bash and execute the commands below. [Git Bash Download Link](https://git-scm.com/downloads/win)

### 1. Clone the Repository and Install Dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/TeamWiseFlow/wiseflow.git
```

Next, download PocketBase for your system from [PocketBase Docs](https://pocketbase.io/docs/) and place it in the `.pb/` folder.

You can also use install_pocketbase.sh (for MacOS/Linux) or install_pocketbase.ps1 (for Windows).

### 2. Configure the .env File

Create a `.env` file in the project root based on `env_sample` and fill in your configurations.

The minimal configuration requires:

*   `LLM_API_KEY=""` (Your LLM service API key)
*   `LLM_API_BASE="https://api.siliconflow.cn/v1"` (LLM service API endpoint; Siliconflow is recommended)
*   `PRIMARY_MODEL=ByteDance-Seed/Seed-OSS-36B-Instruct`
*   `VL_MODEL=Pro/Qwen/Qwen2.5-VL-7B-Instruct`

### 3. Run Wiseflow

```bash
cd wiseflow
uv venv # Run only the first time
source .venv/bin/activate  # Linux/macOS
# or on Windows:
# .venv\Scripts\activate
uv sync # Run only the first time
chmod +x run.sh # Run only the first time
./run.sh
```

For detailed usage instructions, refer to [docs/manual/manual.md](./docs/manual/manual.md).

## Using Scraped Data

Wiseflow stores all scraped data in PocketBase. You can access and integrate data directly through the PocketBase database.

## License

The project is licensed under the new [LICENSE](LICENSE) starting from version 4.2.

For commercial partnerships, please contact **Email: zm.zhao@foxmail.com**.

## Contributing

We welcome contributions! Please submit any questions or suggestions via [issue](https://github.com/TeamWiseFlow/wiseflow/issues).

## Acknowledgements

This project is built upon the following open-source projects:

*   Crawl4ai
*   Patchright
*   MediaCrawler
*   NoDriver
*   Pocketbase
*   Feedparser
*   SearXNG

## Citation

If you use this project in your work, please cite as follows:

```
Author：Wiseflow Team
https://github.com/TeamWiseFlow/wiseflow
```

## Supporting Links

[<img src="docs/logos/SiliconFlow.png" alt="siliconflow" width="360">](https://siliconflow.com/)