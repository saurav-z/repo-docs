# Wiseflow: The AI-Powered Chief Information Officer (CIO)

**Uncover valuable insights from the vast ocean of information with Wiseflow, your AI-powered information filter.**

[View the original repository on GitHub](https://github.com/TeamWiseFlow/wiseflow)

**Key Features:**

*   **AI-Powered Information Filtering:** Wiseflow cuts through the noise of the internet to deliver the information you need.
*   **Customizable Search Sources:** Search across multiple sources including Bing, GitHub, ArXiv, and eBay.
*   **Role-Based Analysis:** Guide the AI with specific roles and objectives for targeted information extraction.
*   **Customizable Extraction Modes:** Create custom forms within the PocketBase interface for precise data extraction.
*   **Social Media Insights:** Identify relevant content and creators on social media platforms.

## 🚀 Enhanced Features in Wiseflow 4.1

Wiseflow 4.1 brings significant improvements, including:

*   **Custom Search Sources:** Configure searches across Bing, GitHub, ArXiv, and eBay using native APIs.
    ![search_source](docs/select_search_source.gif)
*   **Role-Based Analysis:** Direct the AI to analyze information from specific perspectives and goals. See how roles and objectives can influence extraction results in [task1](test/reports/report_v4x_llm/task1).
*   **Custom Extraction Modes:** Create custom forms within the PocketBase interface for precise data extraction.
*   **Social Media Insights:** Find content and creators on social media platforms. Wiseflow can help you find potential customers, partners, or investors by searching social media for relevant content and finding creator information.
    ![find_person_by_wiseflow](docs/find_person_by_wiseflow.png)

**For more details, see the [CHANGELOG](CHANGELOG.md).**

## 🧐 'Wide Search' vs. 'Deep Search'

Wiseflow specializes in "wide search," ideal for broad information gathering (e.g., industry intelligence, background checks) where depth is not required. It's designed for efficiency in these scenarios.

## ✋ What Sets Wiseflow Apart?

*   **Comprehensive Data Sources:** Access information from websites, social media (Weibo, Kuaishou), RSS feeds, and search engines.
*   **Smart HTML Processing:** Automatically extracts relevant information and identifies valuable links.
*   **User-Friendly Design:** No need for manual Xpaths – it's ready to use.
*   **Continuous Improvement:** Expect high stability, availability, and efficient resource management.
*   **More Than Just a Crawler:** Wiseflow is designed to do more.

![4.x full scope](docs/wiseflow4.xscope.png)

## 🌟 Get Started Quickly

Follow these simple steps to start using Wiseflow:

1.  **Download and Install:**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    git clone https://github.com/TeamWiseFlow/wiseflow.git
    ```

    Install PocketBase:

    *   Download from [pocketbase docs](https://pocketbase.io/docs/) and place the executable in the `.pb/` folder.
    *   Alternatively, use the installation scripts:  `install_pocketbase.sh` (macOS/Linux) or `install_pocketbase.ps1` (Windows).

2.  **Configure the .env File:**

    Create a `.env` file in the project root directory, referencing `env_sample`, and fill in the required information.  You'll need:

    *   `LLM_API_KEY=""` (Your LLM API key)
    *   `LLM_API_BASE="https://api.siliconflow.cn/v1"` (Your LLM API endpoint - Consider using [my referral link](https://cloud.siliconflow.cn/i/WNLYbBpi) for SiliconFlow)
    *   `PRIMARY_MODEL=Qwen/Qwen3-14B` (Recommended LLM)
    *   `VL_MODEL=Pro/Qwen/Qwen2.5-VL-7B-Instruct` (Optional, but recommended)

3.  **Run Wiseflow:**

    ```bash
    cd wiseflow
    uv venv # First-time setup
    source .venv/bin/activate  # Linux/macOS
    # Or on Windows:
    # .venv\Scripts\activate
    uv sync # First-time setup
    python -m playwright install --with-deps chromium # First-time setup
    chmod +x run.sh # First-time setup
    ./run.sh
    ```

    For more detailed instructions, refer to [docs/manual/manual.md](./docs/manual/manual.md).

## 📚 Accessing Data

All scraped data is stored in PocketBase. You can use the PocketBase SDKs (Go, Javascript, Python, etc.) to access and work with the data.

Share your projects and applications using Wiseflow data at: [https://github.com/TeamWiseFlow/wiseflow_plus](https://github.com/TeamWiseFlow/wiseflow_plus)

## 🛡️ License

This project is licensed under [Apache2.0](LICENSE).

For commercial collaborations, please contact: **Email：zm.zhao@foxmail.com**

## 📬 Contact

For questions and suggestions, please create an [issue](https://github.com/TeamWiseFlow/wiseflow/issues).

## 🤝 Acknowledgments

Wiseflow is built upon these excellent open-source projects:

*   Crawl4ai: [https://github.com/unclecode/crawl4ai](https://github.com/unclecode/crawl4ai)
*   MediaCrawler: [https://github.com/NanmiCoder/MediaCrawler](https://github.com/NanmiCoder/MediaCrawler)
*   NoDriver: [https://github.com/ultrafunkamsterdam/nodriver](https://github.com/ultrafunkamsterdam/nodriver)
*   Pocketbase: [https://github.com/pocketbase/pocketbase](https://github.com/pocketbase/pocketbase)
*   Feedparser: [https://github.com/kurtmckee/feedparser](https://github.com/kurtmckee/feedparser)
*   SearXNG: [https://github.com/searxng/searxng](https://github.com/searxng/searxng)

## Citation

If you use Wiseflow in your work, please cite it as follows:

```
Author：Wiseflow Team
https://github.com/TeamWiseFlow/wiseflow
Licensed under Apache2.0
```

## 友情链接

[<img src="docs/logos/SiliconFlow.png" alt="siliconflow" width="360">](https://siliconflow.com/)