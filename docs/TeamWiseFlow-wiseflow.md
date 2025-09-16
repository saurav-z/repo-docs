# Wiseflow: Your AI Chief Information Officer 🚀

**[GitHub Repo](https://github.com/TeamWiseFlow/wiseflow)**

**Uncover valuable insights from a sea of information with Wiseflow, an AI-powered tool designed to filter noise and deliver the key takeaways you need.**

---

## Key Features

*   **Enhanced Web Scraping:** Leverage your local Chrome browser for robust and reliable web data extraction.
*   **Customizable Search Sources:** Configure specific search engines (Bing, GitHub, Arxiv) for focused information gathering.
*   **AI-Driven Perspective:** Set roles and objectives for the AI to analyze information from specific viewpoints.
*   **Custom Extraction Forms:** Create tailored forms for precise information extraction based on your needs.
*   **Social Media Discovery:** Find content and creators on social platforms to expand your information sources.
*   **Comprehensive Platform Support:** Access information from websites, social media (Weibo, Kuaishou), RSS feeds, and search engines.
*   **"Wide Search" Focus:** Designed for broad information gathering, ideal for industry intelligence and background research.
*   **PocketBase Integration:** Data is instantly saved to PocketBase, allowing you to access and use the data with ease.

## What's New in Version 4.2

Wiseflow 4.2 builds upon previous versions with a focus on enhanced web scraping capabilities. Key improvements include:

*   **Local Chrome Integration:**  Utilize your local Chrome browser for improved reliability and access to websites requiring login. This eliminates the need for `playwright` installation.
*   **Refreshed Search Engine Implementation:**  Improved and updated search engine integrations.
*   **Complete Proxy Solution:**  Provides a robust proxy solution. See the [CHANGELOG](CHANGELOG.md) for details.

## Model Recommendations

Wiseflow offers flexibility in model selection. Here's a guide to get you started:

*   **Performance:** ByteDance-Seed/Seed-OSS-36B-Instruct
*   **Cost-Effective:** Qwen/Qwen3-14B
*   **Visual Analysis:**  Qwen/Qwen2.5-VL-7B-Instruct

## Getting Started

**Follow these simple steps to start using Wiseflow:**

**Important:**  As of version 4.2, you **must** have Google Chrome installed (using the default installation path).

### 1. Clone the Repository and Install Dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/TeamWiseFlow/wiseflow.git
```

### 2. Configure Your Environment

1.  Download PocketBase from the [Pocketbase Docs](https://pocketbase.io/docs/) and place it in the `.pb/` directory.
2.  Create a `.env` file in the root directory, using `env_sample` as a guide.  At a minimum, you'll need to configure:

    *   `LLM_API_KEY=""`
    *   `LLM_API_BASE="https://api.siliconflow.cn/v1"` (Recommended)
    *   `PRIMARY_MODEL=ByteDance-Seed/Seed-OSS-36B-Instruct`
    *   `VL_MODEL=Pro/Qwen/Qwen2.5-VL-7B-Instruct`

### 3. Run Wiseflow

```bash
cd wiseflow
uv venv # 仅第一次执行需要
source .venv/bin/activate  # Linux/macOS
# 或者在 Windows 上：
# .venv\Scripts\activate
uv sync # 仅第一次执行需要
chmod +x run.sh # 仅第一次执行需要
./run.sh
```

For detailed usage instructions, refer to [docs/manual/manual.md](./docs/manual/manual.md).

## Accessing Your Data

Wiseflow stores all scraped data in PocketBase.  You can use PocketBase's SDKs (Go, Javascript, Python) to access and work with the data.

## Contribute

We welcome contributions!  See the 4.x architecture diagram ([docs/wiseflow4.xscope.png](docs/wiseflow4.xscope.png)) for open development areas. Contributors will receive free access to the Pro version.

## License

*   From version 4.2, please refer to [LICENSE](LICENSE) for the new open source license.
*   For commercial collaborations, please contact: zm.zhao@foxmail.com

## Contact

For questions or suggestions, please open an [issue](https://github.com/TeamWiseFlow/wiseflow/issues).

## Acknowledgements

Wiseflow is built upon several excellent open-source projects:

*   Crawl4ai
*   Patchright
*   MediaCrawler
*   NoDriver
*   Pocketbase
*   Feedparser
*   SearXNG

## Citation

```
Author：Wiseflow Team
https://github.com/TeamWiseFlow/wiseflow
```

## Related Links

[<img src="docs/logos/SiliconFlow.png" alt="siliconflow" width="360">](https://siliconflow.com/)