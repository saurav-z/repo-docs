# Wiseflow: Your AI-Powered Chief Information Officer

**[English](README_EN.md) | [日本語](README_JP.md) | [한국어](README_KR.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [العربية](README_AR.md)**

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TeamWiseFlow/wiseflow)

**Wiseflow uses large language models to cut through the noise and deliver the most important information from a sea of sources, helping you stay ahead of the curve.**

[Link to Original Repo:](https://github.com/TeamWiseFlow/wiseflow)

## Key Features of Wiseflow

*   **Enhanced Web Scraping:** Utilize your local Chrome browser for improved access and data persistence, including login support.
*   **Customizable Search Sources:**  Integrate with Bing, GitHub, and Arxiv for targeted information gathering.
*   **AI-Driven Insights:**  Configure roles and objectives for the LLM to analyze information from a specific perspective.
*   **Custom Extraction Modes:**  Create and apply custom forms within the application for precise data extraction.
*   **Social Media Source Support:**  Find relevant content and creator information on social media platforms.
*   **LLM Recommendations:**  Guidance on selecting the best LLM models based on performance and cost.
*   **Wide Search Focus:**  Specifically designed for broad information gathering, such as industry research and background checks, as opposed to deep, question-specific searches.

## What's New in Wiseflow 4.2?

Wiseflow 4.2 builds on the strengths of previous versions, with a major focus on enhancing web scraping capabilities. This version directly leverages your local Chrome browser, minimizing the risk of being blocked by websites and enabling features like persistent logins and support for page scripts.  Refer to the [CHANGELOG](CHANGELOG.md) for more details.

## Getting Started Quickly

**Follow these three simple steps to begin using Wiseflow:**

**Important:** From version 4.2 onwards, you *must* install Google Chrome (using the default installation path).

### 1.  Download and Install

**Windows users should first install git bash tool, and then execute the following command [bash download link](https://git-scm.com/downloads/win)**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/TeamWiseFlow/wiseflow.git
```
This will install uv.

Then download the corresponding pocketbase program for your system from [pocketbase docs](https://pocketbase.io/docs/) and place it in the [.pb](./pb/) folder.

You can also try using install_pocketbase.sh (for MacOS/Linux) or install_pocketbase.ps1 (for Windows) to install.

### 2. Configure .env File

Create a `.env` file in the root directory of the project, referencing `env_sample`, and fill in the necessary settings.

**Configuration (minimum settings):**
```
LLM_API_KEY=""
LLM_API_BASE="https://api.siliconflow.cn/v1"
PRIMARY_MODEL=ByteDance-Seed/Seed-OSS-36B-Instruct
VL_MODEL=Pro/Qwen/Qwen2.5-VL-7B-Instruct
```

### 3. Run Wiseflow

```bash
cd wiseflow
uv venv # run only the first time
source .venv/bin/activate  # Linux/macOS
# or on Windows:
# .venv\Scripts\activate
uv sync # run only the first time
chmod +x run.sh # run only the first time
./run.sh
```

Refer to [docs/manual/manual.md](./docs/manual/manual.md) for detailed usage instructions.

## Using Your Scraped Data

Wiseflow stores scraped data directly in PocketBase.  You can access and utilize this data by interacting with the PocketBase database directly.  SDKs are available in Go, Javascript, and Python.

##  Contribute and Connect

*   **Issue submissions and suggestions:**  [Issues](https://github.com/TeamWiseFlow/wiseflow/issues)
*   **Commercial Partnerships:** Contact zm.zhao@foxmail.com
*   **Community Projects:** Share your secondary development applications at https://github.com/TeamWiseFlow/wiseflow_plus

## License

From version 4.2, the open-source license has been updated.  Please review the [LICENSE](LICENSE) file.

## Acknowledgements

Wiseflow leverages these excellent open-source projects:

*   Crawl4ai: [https://github.com/unclecode/crawl4ai](https://github.com/unclecode/crawl4ai)
*   Patchright: [https://github.com/Kaliiiiiiiiii-Vinyzu/patchright-python](https://github.com/Kaliiiiiiiiii-Vinyzu/patchright-python)
*   MediaCrawler: [https://github.com/NanmiCoder/MediaCrawler](https://github.com/NanmiCoder/MediaCrawler)
*   NoDriver: [https://github.com/ultrafunkamsterdam/nodriver](https://github.com/ultrafunkamsterdam/nodriver)
*   Pocketbase: [https://github.com/pocketbase/pocketbase](https://github.com/pocketbase/pocketbase)
*   Feedparser: [https://github.com/kurtmckee/feedparser](https://github.com/kurtmckee/feedparser)
*   SearXNG: [https://github.com/searxng/searxng](https://github.com/searxng/searxng)

## Citation

If you use or reference Wiseflow in your work, please cite it as:

```
Author：Wiseflow Team
https://github.com/TeamWiseFlow/wiseflow
```

## Recommended Resources

[<img src="docs/logos/SiliconFlow.png" alt="siliconflow" width="360">](https://siliconflow.com/)