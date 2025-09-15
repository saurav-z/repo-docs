# Wiseflow: Your AI-Powered Chief Information Officer

**Uncover hidden insights from vast information sources and stay ahead with Wiseflow, your AI-powered information retrieval and analysis tool. [Visit the original repository](https://github.com/TeamWiseFlow/wiseflow)**

Wiseflow cuts through the noise, delivering the key information you need by extracting essential insights from a sea of data.

## Key Features

*   ✅ **Web Scraping 2.0:** Enhanced web-scraping capabilities leverage your local Chrome browser for improved reliability and persistent data access.
*   ✅ **Custom Search Sources:** Configure and fine-tune your search with integrated support for Bing, GitHub, and Arxiv.
*   ✅ **Role-Based Analysis:** Guide the LLM with custom roles and objectives to analyze information from specific perspectives.
*   ✅ **Custom Extraction Forms:** Create custom forms to structure and precisely extract the data you need.
*   ✅ **Social Media Discovery:** Find content and creators on social media platforms, assisting in lead generation and networking.
*   ✅ **Optimized LLM Integration:** Recommendations for the best LLM models based on performance and cost, offering flexibility in choosing your AI backend.
*   ✅ **"Wide Search" Focused:** Designed for broad information gathering across various sources, unlike "deep search" tools that focus on specific questions.

## What's New in Version 4.2

Wiseflow 4.2 introduces significant upgrades, including:

*   **Enhanced Web Scraping:**  Leverages your local Chrome browser to reduce the risk of detection and provide access to logged-in content.
*   **Refactored Search Engine and Proxy Support:** Improved search engine infrastructure for enhanced search capabilities, and a complete proxy solution.
*   **No More Playwright Installation:** Removes the need to install `playwright` dependencies during setup, making deployment smoother.

For detailed changes, see the [CHANGELOG](CHANGELOG.md).

## Getting Started

1.  **Prerequisites:** Install Google Chrome (using the default installation path) and git bash (for Windows users).
2.  **Clone and Install:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    git clone https://github.com/TeamWiseFlow/wiseflow.git
    ```
3.  **Configure `.env`:** Create a `.env` file based on the `env_sample` and fill in the required settings (LLM API key, LLM API base, primary and vision models).
4.  **Run Wiseflow:**
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

Refer to [docs/manual/manual.md](./docs/manual/manual.md) for comprehensive usage instructions.

## Data Access & Integration

Wiseflow stores all scraped data in PocketBase. Use PocketBase SDKs (available for Go, Javascript, and Python) to access and integrate the data into your applications.

Share your custom applications and developments here: [https://github.com/TeamWiseFlow/wiseflow_plus](https://github.com/TeamWiseFlow/wiseflow_plus)

## License and Contact

*   **License:**  The project is released under a new open-source license, available in [LICENSE](LICENSE).
*   **Commercial inquiries:** Contact zm.zhao@foxmail.com.
*   **Feedback and issues:** Please report any issues or suggestions via [GitHub issues](https://github.com/TeamWiseFlow/wiseflow/issues).

## Acknowledgements

Wiseflow is built on the following excellent open-source projects:
*   Crawl4ai
*   Patchright
*   MediaCrawler
*   NoDriver
*   Pocketbase
*   Feedparser
*   SearXNG

## Citation

If you use or reference Wiseflow in your work, please cite the following:

```
Author：Wiseflow Team
https://github.com/TeamWiseFlow/wiseflow
```

## Featured Partner

[<img src="docs/logos/SiliconFlow.png" alt="siliconflow" width="360">](https://siliconflow.com/)