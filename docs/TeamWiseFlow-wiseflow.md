# Wiseflow: Your AI-Powered Information Navigator

**Tired of information overload? Wiseflow sifts through the noise, delivering the key insights you need from vast amounts of data.**

[Visit the original repository](https://github.com/TeamWiseFlow/wiseflow)

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TeamWiseFlow/wiseflow)

## Key Features

*   **AI-Driven Information Extraction:**  Leverage large language models (LLMs) to automatically extract relevant information from web pages, social media, and more.
*   **Customizable Search Sources:** Integrate with Bing, GitHub, arXiv, and other platforms to focus your searches.
*   **Contextual Analysis with AI:** Define roles and goals for the LLM to tailor analysis and extraction to your specific needs.
*   **Customizable Extraction Templates:**  Create custom forms to extract specific data points from web pages.
*   **Social Media Source Discovery:**  Identify content and creators on social platforms based on your focus points.
*   **Local Chrome Browser Integration:** Version 4.2 and later directly uses your local Chrome browser to enhance web scraping capabilities, bypass anti-bot measures, and support user logins for access to content requiring authentication.

## What's New in Version 4.2

*   **Enhanced Web Scraping:**  Improved web crawling by directly integrating with your local Chrome browser.
*   **Refactored Search Engine:** Includes new search engine options.
*   **Comprehensive Proxy Support:** Offers robust proxy solutions for reliable access.
*   **No Playwright Dependency:** Removes the need to install Playwright dependencies for deployment.
*   **More Stable and Reliable:**  Continuously improved for stability and availability.

For details, see the [CHANGELOG](CHANGELOG.md).

## LLM Recommendation

Based on extensive testing, the following models are recommended for optimal performance and cost-effectiveness:

*   **Performance-Focused:** `ByteDance-Seed/Seed-OSS-36B-Instruct`
*   **Cost-Effective:** `Qwen/Qwen3-14B`
*   **Visual Analysis:**  `Qwen/Qwen2.5-VL-7B-Instruct` (for visual analysis tasks)

Detailed test reports are available at [LLM USE TEST](./test/reports/README.md).

Wiseflow is compatible with OpenAI SDK-compatible LLM services, allowing you to use various providers such as Siliconflow, local Ollama deployments, or other options.

## Deep Search vs. Wide Search

Wiseflow is designed for "wide search" scenarios, offering a cost-effective solution for broad information gathering, such as industry intelligence or background checks.  It avoids the complexity and cost of "deep search" approaches when broad information gathering is needed.

## Getting Started

**Follow these three steps to start using Wiseflow:**

**Important:**  Starting with version 4.2, you must have Google Chrome installed on your system (using the default installation path).

### 1.  Clone the Repository and Install Dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/TeamWiseFlow/wiseflow.git
```

### 2. Install PocketBase

Download the appropriate PocketBase binary from [PocketBase docs](https://pocketbase.io/docs/) and place it in the `.pb` folder. You can also try to use:

* install\_pocketbase.sh (for MacOS/Linux)
* install\_pocketbase.ps1 (for Windows) to install.

### 3. Configure the Environment

Create a `.env` file in the root directory of the project (based on `env_sample`) and configure the following parameters:

*   `LLM_API_KEY=""` (Your LLM service API key.  Leave blank if using a local Ollama deployment.)
*   `LLM_API_BASE="https://api.siliconflow.cn/v1"` (LLM API endpoint. Recommended: Siliconflow.  See the original repo for a referral link.)
*   `PRIMARY_MODEL=ByteDance-Seed/Seed-OSS-36B-Instruct` (The primary LLM model to be used.)
*   `VL_MODEL=Pro/Qwen/Qwen2.5-VL-7B-Instruct` (Visual LLM model.)

### 4. Run Wiseflow

```bash
cd wiseflow
uv venv # Run this command only the first time
source .venv/bin/activate  # Linux/macOS
# Or on Windows:
# .venv\Scripts\activate
uv sync # Run this command only the first time
chmod +x run.sh # Run this command only the first time
./run.sh
```

For a detailed guide, consult [docs/manual/manual.md](./docs/manual/manual.md).

## Integrating Data

All scraped data is stored in PocketBase, so you can directly access it through the database using a Go/Javascript/Python SDK.

## Contributing

The Wiseflow project welcomes contributions.  Check out the [4.x architecture diagram](docs/wiseflow4.xscope.png) to see the project's future plans.  Contributors may receive free access to the Pro version.

## License

[LICENSE](LICENSE)

For commercial partnerships, please contact **zm.zhao@foxmail.com**.

## Contact

For any questions or suggestions, please open an [issue](https://github.com/TeamWiseFlow/wiseflow/issues).

## Acknowledgments

Wiseflow is built upon these excellent open-source projects:

*   Crawl4ai ([https://github.com/unclecode/crawl4ai](https://github.com/unclecode/crawl4ai))
*   Patchright ([https://github.com/Kaliiiiiiiiii-Vinyzu/patchright-python](https://github.com/Kaliiiiiiiiii-Vinyzu/patchright-python))
*   MediaCrawler ([https://github.com/NanmiCoder/MediaCrawler](https://github.com/NanmiCoder/MediaCrawler))
*   NoDriver ([https://github.com/ultrafunkamsterdam/nodriver](https://github.com/ultrafunkamsterdam/nodriver))
*   Pocketbase ([https://github.com/pocketbase/pocketbase](https://github.com/pocketbase/pocketbase))
*   Feedparser ([https://github.com/kurtmckee/feedparser](https://github.com/kurtmckee/feedparser))
*   SearXNG ([https://github.com/searxng/searxng](https://github.com/searxng/searxng))

## Citation

If you use this project in your work, please cite it as follows:

```
Author：Wiseflow Team
https://github.com/TeamWiseFlow/wiseflow
```

## Resources

[<img src="docs/logos/SiliconFlow.png" alt="siliconflow" width="360">](https://siliconflow.com/)