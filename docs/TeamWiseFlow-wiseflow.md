# Wiseflow: Your AI-Powered Chief Intelligence Officer

**[English](README_EN.md) | [日本語](README_JP.md) | [한국어](README_KR.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [العربية](README_AR.md)**

**Tired of information overload? Wiseflow is an AI-powered tool that filters the noise and delivers the key insights you need from vast amounts of data and diverse sources. [Visit the original repository](https://github.com/TeamWiseFlow/wiseflow) for more information.**

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TeamWiseFlow/wiseflow)

## Key Features

*   **Comprehensive Data Sources:** Access information from webpages, social media (Weibo, Kuaishou), RSS feeds, and search engines like Bing, GitHub, and Arxiv.
*   **AI-Driven Filtering:** Leverage large language models (LLMs) to extract and prioritize relevant information based on your specific focus points.
*   **Customizable Extraction:** Create custom forms for precise data extraction and tailor LLM analysis with roles and objectives.
*   **Social Media Insights:**  Find content creators and potential contacts on social media platforms.
*   **Chrome Integration:** Utilize your local Chrome browser for enhanced web scraping capabilities, bypassing site restrictions.
*   **Flexible LLM Compatibility:** Works with any LLM that supports the OpenAI API format, including local deployments.
*   **Wide Search Focus:** Designed for broad information gathering across various sources, unlike "deep search" approaches.
*   **PocketBase Integration:** Data is stored in PocketBase for easy access and integration into your own applications.

## What's New in Version 4.2

*   **Enhanced Web Scraping:** Direct integration with your local Chrome browser for improved data acquisition.
*   **Customizable Search Sources:** Configure search sources (Bing, GitHub, Arxiv) for focused information retrieval.
*   **Improved Performance and Stability:**  Refactored search engine and robust proxy solutions.

## Getting Started

**Installation is simple, follow these steps:**

**Prerequisites:**

1.  **Install Google Chrome:** Ensure Chrome is installed in the default directory.
2.  **Windows Users:** Install Git Bash.

### 1. Clone the Repository & Install Dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/TeamWiseFlow/wiseflow.git
```

### 2. Install Pocketbase

Download the appropriate Pocketbase program for your system from the [Pocketbase Docs](https://pocketbase.io/docs/) and place it in the `.pb/` directory.

You can also use `install_pocketbase.sh` (MacOS/Linux) or `install_pocketbase.ps1` (Windows).

### 3. Configure the `.env` File

Create a `.env` file in the project root based on the `env_sample` template.  At minimum, you need to provide:

-   `LLM_API_KEY=""` - Your LLM service API key.
-   `LLM_API_BASE="https://api.siliconflow.cn/v1"` - LLM service API endpoint.
-   `PRIMARY_MODEL=ByteDance-Seed/Seed-OSS-36B-Instruct` - Primary LLM.
-   `VL_MODEL=Pro/Qwen/Qwen2.5-VL-7B-Instruct` - Vision-Language Model

### 4. Run Wiseflow

```bash
cd wiseflow
uv venv # Run only the first time
source .venv/bin/activate  # Linux/macOS
# Or on Windows:
# .venv\Scripts\activate
uv sync # Run only the first time
chmod +x run.sh # Run only the first time
./run.sh
```

For detailed usage instructions, see [docs/manual/manual.md](./docs/manual/manual.md).

## Integrate with Your Applications

All scraped data is stored in Pocketbase. Use the PocketBase SDKs (Go/Javascript/Python) to access the data directly.

## Contribute

We welcome contributions!  See the [4.x scope diagram](docs/wiseflow4.xscope.png) for areas needing development. Contributors will receive free access to the pro version.

## License

See the [LICENSE](LICENSE) for the updated open-source license. For commercial collaborations, contact zm.zhao@foxmail.com.

## Contact

For questions or suggestions, please open an [issue](https://github.com/TeamWiseFlow/wiseflow/issues).

## Acknowledgements

This project is built upon the following open-source projects: (list of open-source projects from original README).

## Citation

Please cite the project as follows if you use it in your work:

```
Author：Wiseflow Team
https://github.com/TeamWiseFlow/wiseflow
```

## Related Resources

*   [SiliconFlow](https://siliconflow.com/)