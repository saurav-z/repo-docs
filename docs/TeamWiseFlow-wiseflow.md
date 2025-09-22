# Wiseflow: Your AI-Powered Chief Intelligence Officer

**Tired of information overload? Wiseflow uses large language models (LLMs) to filter the noise and deliver the most valuable insights from vast data sources.** ([Original Repo](https://github.com/TeamWiseFlow/wiseflow))

## Key Features:

*   **Advanced Web Scraping:** Capture data directly from websites using your local Chrome browser.
*   **Customizable Search Sources:** Utilize Bing, GitHub, and Arxiv search engines.
*   **Role-Based Analysis:** Guide LLMs with roles and objectives for targeted information extraction.
*   **Custom Extraction Forms:** Create tailored forms for precise data retrieval.
*   **Social Media Content Discovery:** Uncover relevant content and creators on social platforms.
*   **LLM Recommendation:** Get suggestions for the best LLMs for your specific tasks.
*   **PocketBase Integration:** Seamlessly access scraped data through PocketBase.
*   **Wide Search Focus:** Efficiently gather broad information for tasks like industry analysis and customer research.

## What's New in Wiseflow 4.2:

*   **Enhanced Web Scraping:** Leverage your local Chrome for more reliable and customizable web data extraction.
*   **Simplified Deployment:** No more Playwright installation needed.
*   **Improved Search Engine and Proxy Solutions:** Enhanced system performance and flexibility.

## Getting Started:

1.  **Install Chrome:** Ensure Google Chrome is installed using the default installation path.
2.  **Clone the Repository:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    git clone https://github.com/TeamWiseFlow/wiseflow.git
    ```
3.  **Configure .env:** Create a `.env` file (based on `env_sample`) with your LLM API key, base URL, and model selections.
4.  **Run Wiseflow:**
    ```bash
    cd wiseflow
    uv venv # (First time only)
    source .venv/bin/activate  # Linux/macOS
    # Or on Windows: .venv\Scripts\activate
    uv sync # (First time only)
    chmod +x run.sh # (First time only)
    ./run.sh
    ```

    Detailed usage instructions are available in [docs/manual/manual.md](./docs/manual/manual.md).

## Data Access:

Scraped data is stored in PocketBase. Use PocketBase SDKs (Go/Javascript/Python) to access and utilize the extracted information.

## Contribute:

Explore additional applications at https://github.com/TeamWiseFlow/wiseflow_plus.

## License:

See [LICENSE](LICENSE) for the updated open-source license (version 4.2 onwards).

## Contact:

For questions or suggestions, please open an [issue](https://github.com/TeamWiseFlow/wiseflow/issues).

## Acknowledgements:

Wiseflow is built upon several exceptional open-source projects:

*   [Crawl4ai](https://github.com/unclecode/crawl4ai)
*   [Patchright](https://github.com/Kaliiiiiiiiii-Vinyzu/patchright-python)
*   [MediaCrawler](https://github.com/NanmiCoder/MediaCrawler)
*   [NoDriver](https://github.com/ultrafunkamsterdam/nodriver)
*   [Pocketbase](https://github.com/pocketbase/pocketbase)
*   [Feedparser](https://github.com/kurtmckee/feedparser)
*   [SearXNG](https://github.com/searxng/searxng)

## Citation:

Please cite Wiseflow if you use it in your work:

```
Author：Wiseflow Team
https://github.com/TeamWiseFlow/wiseflow
```

## Collaboration:

For commercial partnerships, please contact zm.zhao@foxmail.com.