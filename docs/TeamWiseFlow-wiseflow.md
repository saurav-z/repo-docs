# Wiseflow: Your AI-Powered Information Navigator

**Tired of information overload? Wiseflow leverages large language models to filter the noise and deliver the essential insights you need from vast information sources. ([Original Repo](https://github.com/TeamWiseFlow/wiseflow))**

**[English](README_EN.md) | [日本語](README_JP.md) | [한국어](README_KR.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [العربية](README_AR.md)**

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TeamWiseFlow/wiseflow)

## Key Features

*   **Intelligent Information Extraction:**  Uses LLMs to extract key information from a wide range of sources.
*   **Web Scraping with Enhanced Capabilities:** Includes a new feature in version 4.2 that enables local Chrome browser integration to bypass website security measures and allow user logins.
*   **Customizable Search Sources:**  Supports Bing, GitHub, and Arxiv with native APIs, eliminating the need for third-party services.
*   **AI-Driven Perspective:** Allows setting roles and objectives for LLMs to analyze information from specific viewpoints.
*   **Custom Extraction Forms:** Enables the creation of custom forms to extract specific data points.
*   **Social Media Source Search:** Identifies content and creators on social media platforms.
*   **Flexible LLM Integration:** Compatible with various LLM providers and local deployment options.

## What's New in Wiseflow 4.2

Wiseflow 4.2 features enhanced web crawling abilities using your local Chrome browser, improving reliability and introducing new features like persistent user data and scripting capabilities.  Other notable updates include a refactored search engine and comprehensive proxy solutions.

## LLM Recommendation

Wiseflow has tested and recommends these LLMs based on performance and cost:

*   **Performance-focused:** ByteDance-Seed/Seed-OSS-36B-Instruct
*   **Cost-effective:** Qwen/Qwen3-14B
*   **Visual Analysis (optional):** /Qwen/Qwen2.5-VL-7B-Instruct

See [LLM USE TEST](./test/reports/README.md) for detailed test results.

## 'Wide Search' vs. 'Deep Search'

Wiseflow is designed for "wide search" scenarios. Unlike "deep search" methods that focus on a single question, Wiseflow efficiently gathers broad information for tasks like industry intelligence, background checks, and lead generation.

## Why Choose Wiseflow?

*   **Comprehensive Source Support:** Web pages, social media (Weibo, Kuaishou), RSS feeds, and more.
*   **Smart HTML Processing:** Extracts relevant information and identifies links for further exploration.
*   **Integrated Crawling & LLM:** LLMs analyze during the crawl, minimizing platform detection.
*   **User-Friendly:** Designed for ease of use without requiring manual Xpath input.
*   **Continuously Improved:**  High stability and efficiency with each iteration.
*   **Beyond Crawling:** Wiseflow is designed as a powerful information discovery tool.

<img src="docs/wiseflow4.xscope.png" alt="4.x full scope" width="720">

(4.x architecture plan.  Parts within the dashed lines are in development.  Community contributions are welcome!)

## Getting Started

**Follow these three steps to begin!**

**Google Chrome installation (using the default installation path) is required starting with version 4.2.**

**Windows users, install the git bash tool and run the following commands inside it. [bash download](https://git-scm.com/downloads/win)**

### 1. Download and Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/TeamWiseFlow/wiseflow.git
```

This installs `uv`.

Next, download the PocketBase program from [pocketbase docs](https://pocketbase.io/docs/) for your system and place it in the [.pb](./pb/) folder.

Alternatively, try using install_pocketbase.sh (for MacOS/Linux) or install_pocketbase.ps1 (for Windows).

### 2. Configure the Environment

Create a `.env` file in the Wiseflow directory (project root) using `env_sample` as a reference. Minimum configuration includes:

*   `LLM_API_KEY=""`  (LLM API key)
*   `LLM_API_BASE="https://api.siliconflow.cn/v1"` (LLM API endpoint)
*   `PRIMARY_MODEL=ByteDance-Seed/Seed-OSS-36B-Instruct` (Primary LLM)
*   `VL_MODEL=Pro/Qwen/Qwen2.5-VL-7B-Instruct` (Visual LLM)

### 3. Run Wiseflow

```bash
cd wiseflow
uv venv # Run once
source .venv/bin/activate  # Linux/macOS
# Or on Windows:
# .venv\Scripts\activate
uv sync # Run once
chmod +x run.sh # Run once
./run.sh
```

See [docs/manual/manual.md](./docs/manual/manual.md) for detailed instructions.

## Accessing Data

All scraped data is stored in PocketBase. Use PocketBase SDKs for Go, Javascript, or Python to access the data.

## Community & Contributions

Share your secondary development application cases in:

-   https://github.com/TeamWiseFlow/wiseflow_plus

## License

From version 4.2, the project uses a new open-source license: [LICENSE](LICENSE)

For commercial collaborations, please contact **Email：zm.zhao@foxmail.com**

## Contact

For questions or suggestions, please create an [issue](https://github.com/TeamWiseFlow/wiseflow/issues).

## Acknowledgements

This project utilizes these excellent open-source projects:

*   Crawl4ai, Patchright, MediaCrawler, NoDriver, Pocketbase, Feedparser, SearXNG

## Citation

```
Author：Wiseflow Team
https://github.com/TeamWiseFlow/wiseflow
```

## Resources

[<img src="docs/logos/SiliconFlow.png" alt="siliconflow" width="360">](https://siliconflow.com/)