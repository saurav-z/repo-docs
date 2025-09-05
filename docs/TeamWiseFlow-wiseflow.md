# Wiseflow: Your AI Chief Intelligence Officer

**Tired of drowning in information?** Wiseflow is an AI-powered tool designed to extract and deliver the most relevant information from a vast sea of sources, saving you time and uncovering valuable insights. ([Original Repository](https://github.com/TeamWiseFlow/wiseflow))

**Key Features:**

*   **Comprehensive Information Gathering:** Collects data from a variety of sources, including web pages, social media platforms (Weibo, Kuaishou), RSS feeds, and search engines (Bing, GitHub, Arxiv, eBay).
*   **AI-Driven Information Filtering:** Uses large language models (LLMs) to filter noise and highlight valuable information based on your specific interests.
*   **Customizable Search Sources:** Choose from a selection of search engines for focused information gathering.
*   **Role-Based Analysis:** Instructs LLMs to analyze information from a specific perspective or for a particular purpose.
*   **Custom Extraction Modes:** Create custom forms within the PocketBase interface to extract specific data fields.
*   **Social Media Creator Discovery:** Identifies content creators related to your focus points on social media.
*   **90% Discount on OpenAI Models:** Enjoy discounted access to OpenAI models through the AiHubMix service (requires switching to the aihubmix branch).

## What Makes Wiseflow Different?

*   **Broad Search Focus:** Designed for "wide search" scenarios, where you need to gather a wide range of information rather than a deep dive into a specific query.
*   **User-Friendly:** Operates without requiring manual Xpath inputs, making it accessible to users without coding expertise.
*   **Adaptive Processing:** Employs a unique HTML processing flow optimized for information extraction.
*   **Stable and Efficient:** Continuously updated for high stability and performance, while being efficient with system resources.

<img src="docs/wiseflow4.xscope.png" alt="4.x full scope" width="720">

*(4.x Architecture Overview. Dashed boxes indicate features under development.  Contributions are welcome!)*

## Getting Started Quickly

**Follow these simple steps to get started:**

**Prerequisites:**

*   Windows users, download and install Git Bash: [Git Bash Download](https://git-scm.com/downloads/win)

### 1. Clone the Repository & Install Dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/TeamWiseFlow/wiseflow.git
```

### 2. Download PocketBase

Download the PocketBase application for your operating system from the [PocketBase Docs](https://pocketbase.io/docs/) and place it in the `.pb/` folder.

*Alternatively*, use the provided scripts (install\_pocketbase.sh for MacOS/Linux or install\_pocketbase.ps1 for Windows) to install PocketBase.

### 3. Configure the .env File

Create a `.env` file in the project's root directory using `env_sample` as a template, and fill in the required settings:

*   `LLM_API_KEY=""` (Your LLM service API key.  Use a service that provides OpenAI-compatible API, or leave blank if deploying locally with Ollama.)
*   `LLM_API_BASE="https://api.siliconflow.cn/v1"` (LLM API endpoint.  Consider using SiliconFlow, and use this [referral link](https://cloud.siliconflow.cn/i/WNLYbBpi) for a sign-up bonus!)
*   `PRIMARY_MODEL=Qwen/Qwen3-14B` (Recommended LLM model)
*   `VL_MODEL=Pro/Qwen/Qwen2.5-VL-7B-Instruct` (Recommended Visual Language Model)

### 4. Run Wiseflow

```bash
cd wiseflow
uv venv # Only required the first time
source .venv/bin/activate  # Linux/macOS
# Or on Windows:
# .venv\Scripts\activate
uv sync # Only required the first time
python -m playwright install --with-deps chromium # Only required the first time
chmod +x run.sh # Only required the first time
./run.sh
```

For more detailed instructions, see [docs/manual/manual.md](./docs/manual/manual.md).

## Accessing the Data

All data extracted by Wiseflow is stored in PocketBase. You can directly access and utilize this data using the PocketBase SDKs available for various programming languages (Go, JavaScript, Python, etc.).

## Contributing

We encourage you to share and promote your secondary development applications in the following repository.

*   https://github.com/TeamWiseFlow/wiseflow\_plus

## License

This project is licensed under the [Apache 2.0](LICENSE) license.

For commercial collaborations, contact **Email: zm.zhao@foxmail.com**

## Contact

For any questions or suggestions, please submit an issue via [issue](https://github.com/TeamWiseFlow/wiseflow/issues).

## Acknowledgements

Wiseflow is built upon these excellent open-source projects:

*   Crawl4ai (Open-source LLM Friendly Web Crawler & Scraper) https://github.com/unclecode/crawl4ai
*   MediaCrawler (xhs/dy/wb/ks/bilibili/zhihu crawler) https://github.com/NanmiCoder/MediaCrawler
*   NoDriver (Providing a blazing fast framework for web automation, webscraping, bots and any other creative ideas...) https://github.com/ultrafunkamsterdam/nodriver
*   Pocketbase (Open Source realtime backend in 1 file) https://github.com/pocketbase/pocketbase
*   Feedparser (Parse feeds in Python) https://github.com/kurtmckee/feedparser
*   SearXNG (a free internet metasearch engine which aggregates results from various search services and databases) https://github.com/searxng/searxng

## Citation

If you reference or use this project in your work, please cite it as follows:

```
Author：Wiseflow Team
https://github.com/TeamWiseFlow/wiseflow
Licensed under Apache2.0
```

## Resources

[<img src="docs/logos/SiliconFlow.png" alt="siliconflow" width="360">](https://siliconflow.com/)