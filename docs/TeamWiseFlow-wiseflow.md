# Wiseflow: Your AI-Powered Chief Intelligence Officer

**[English](README_EN.md) | [日本語](README_JP.md) | [한국어](README_KR.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [العربية](README_AR.md)**

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TeamWiseFlow/wiseflow)

**Wiseflow helps you cut through the noise and uncover crucial insights from the vast ocean of information, using advanced AI to gather, filter, and deliver what matters most to you.**

[Original Repository](https://github.com/TeamWiseFlow/wiseflow)

## Key Features

*   **Web Scraping:** Efficiently extracts data from various sources, including websites, social media platforms (Weibo, Kuaishou), RSS feeds, and search engines.
*   **Customizable Search:** Configure search sources like Bing, GitHub, and arXiv to precisely target your information needs.
*   **AI-Driven Insights:** Leverage LLMs to analyze and extract key information, enabling you to define roles and objectives for tailored analysis.
*   **Custom Extraction:** Create custom forms to structure data extraction and find potential clients, partners, or investors via social media.
*   **Chrome Integration:** Version 4.2 uses your local Chrome browser to enhance scraping capabilities, reduce detection, and support persistent logins.
*   **Flexible LLM Compatibility:** Works with various LLMs compatible with the OpenAI API format, including local deployments via Ollama.

## What's New in Wiseflow 4.2?

Wiseflow 4.2 enhances web scraping and offers:

*   **Local Chrome Browser Integration:** Improved scraping with your local Chrome browser.
*   **Enhanced Search Engine Options:** Improved search engine support
*   **Proxy Support:** Provides a complete proxy solution.

## Why Choose Wiseflow?

Wiseflow stands out from the crowd with its:

*   **Comprehensive Source Coverage:** From web pages to social media, Wiseflow gathers data from many sources.
*   **Intelligent HTML Processing:** LLMs analyze the content and prioritize relevant information.
*   **Adaptive AI Approach:** LLMs are integrated throughout the data gathering process, to help stay under the radar of platform monitoring.
*   **User-Friendly Design:** Designed for ease of use without requiring developer-level expertise.
*   **Continuous Improvement:** High stability, efficiency, and system resource management through continuous updates.
*   **Focus on Wide Search:** Ideal for broad information gathering, unlike the narrow focus of deep search tools.

## Getting Started

**Follow these three steps to start using Wiseflow:**

**Important:**  You must install Google Chrome using the default installation path starting with version 4.2.

### 1. Clone the Repository & Install Dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/TeamWiseFlow/wiseflow.git
```

### 2. Set Up PocketBase

Download the PocketBase program for your operating system from [PocketBase Docs](https://pocketbase.io/docs/) and place it in the `.pb/` directory.

You can also use `install_pocketbase.sh` (MacOS/Linux) or `install_pocketbase.ps1` (Windows).

### 3. Configure the .env File

Create a `.env` file in the project root using the provided `env_sample` as a template.  Configure the following settings:

```
LLM_API_KEY="" # LLM API key (any provider with OpenAI API format; not needed for local Ollama)
LLM_API_BASE="https://api.siliconflow.cn/v1" # LLM API endpoint (recommended: Siliconflow)
PRIMARY_MODEL=ByteDance-Seed/Seed-OSS-36B-Instruct # Alternative: Qwen3-14B for cost-effectiveness
VL_MODEL=Pro/Qwen/Qwen2.5-VL-7B-Instruct
```

### 4. Run Wiseflow

```bash
cd wiseflow
uv venv # Only required the first time
source .venv/bin/activate  # Linux/macOS
# Or on Windows:
# .venv\Scripts\activate
uv sync # Only required the first time
chmod +x run.sh # Only required the first time
./run.sh
```

Refer to [docs/manual/manual.md](./docs/manual/manual.md) for detailed usage instructions.

## Integrating with Your Applications

Access the scraped data in PocketBase by interacting with the database.  PocketBase offers SDKs in Go, JavaScript, and Python.

Share your applications in the [wiseflow_plus](https://github.com/TeamWiseFlow/wiseflow_plus) repository.

## License & Contact

*   **License:** [LICENSE](LICENSE)
*   **Commercial Collaboration:**  Contact zm.zhao@foxmail.com
*   **Issues/Suggestions:** [Issues](https://github.com/TeamWiseFlow/wiseflow/issues)

## Acknowledgements

Wiseflow is built on the following open-source projects: (list of projects from original README)

## Citation

Please cite Wiseflow if used in your projects:

```
Author：Wiseflow Team
https://github.com/TeamWiseFlow/wiseflow
```

## Related Links

[<img src="docs/logos/SiliconFlow.png" alt="siliconflow" width="360">](https://siliconflow.com/)