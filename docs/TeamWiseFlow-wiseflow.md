# Wiseflow: Your AI-Powered Chief Intelligence Officer

**[Original Repository](https://github.com/TeamWiseFlow/wiseflow)**

**Tired of information overload? Wiseflow uses cutting-edge AI to sift through vast amounts of data and uncover the insights you truly need.**

---

## Key Features:

*   **Customizable Search Sources:** Pinpoint your research with support for Bing, GitHub, Arxiv, and eBay, using native APIs.
*   **AI-Driven Analysis with Context:**  Guide the AI with roles and objectives to tailor analysis and extraction.
*   **Customizable Extraction Modes:** Create custom forms within the PocketBase interface for precise data extraction.
*   **Social Media Source Support:** Discover content and creator profiles on social platforms, perfect for lead generation and networking.
*   **Wide Search Focused:** Optimized for broad information gathering, perfect for industry intelligence, background checks, and lead generation.
*   **Platform Agnostic:** Gather information from a variety of platforms including web pages, social media, RSS feeds, and more.

---

## 💰 Discounted OpenAI Access

Enjoy a 10% discount on all OpenAI models through the Wiseflow application (via the AihubMix service).

*   **Important:** Switch to the `aihubmix` branch for discount access. See [README](https://github.com/TeamWiseFlow/wiseflow/blob/aihubmix/README.md).

---

## 🔥 What's New in Wiseflow 4.1?

This release brings exciting new features to improve your information gathering:

### Custom Search Sources

Configure search sources for precise information gathering. Current sources include Bing, Github, Arxiv, and eBay.

<img src="docs/select_search_source.gif" alt="search_source" width="360">

### AI-Driven Analysis with Context

Define roles and objectives to guide the LLM's perspective.

*   **Note:** The impact of roles and objectives is more significant when your focus is less specific.

### Customizable Extraction Modes

Create forms and configure them for specific focus points for tailored information extraction.

### Social Media Source Support

Discover content and creator profiles on social platforms.

<img src="docs/find_person_by_wiseflow.png" alt="find_person_by_wiseflow" width="720">

**For a detailed list of updates, see the [CHANGELOG](CHANGELOG.md).**

---

## 🧐 'Wide Search' vs. 'Deep Search'

Wiseflow is designed for "wide search," focusing on broad information gathering. This is ideal for tasks like industry intelligence and lead generation, offering a more efficient alternative to "deep search" for such purposes.

---

## ✋ What Makes Wiseflow Different?

*   **Comprehensive Source Support:** Web, social media, RSS, and search engines.
*   **Smart HTML Processing:** Extracts key information and identifies further exploration links.
*   **User-Friendly:** No Xpath or manual configuration required.
*   **Stable and Efficient:** Continuously updated for optimal performance.
*   **Evolving Capabilities:** More than just a crawler.

<img src="docs/wiseflow4.xscope.png" alt="4.x full scope" width="720">

(4.x architecture diagram; dotted boxes are incomplete. Contributions welcome!)

---

## 🌟 Quick Start: Get Started in 3 Steps!

**Windows users, install Git Bash first.  Then, run the following in Git Bash:**

### 📋 Download and Install Dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/TeamWiseFlow/wiseflow.git
```

Next, download the appropriate [PocketBase](https://pocketbase.io/docs/) program and place it in the `.pb/` directory.  You can also use:

*   `install_pocketbase.sh` (MacOS/Linux)
*   `install_pocketbase.ps1` (Windows)

### 📥 Configure the .env File

Create a `.env` file in the root directory, referencing `env_sample`, and configure your settings. You'll need these parameters:

*   `LLM_API_KEY=""`
*   `LLM_API_BASE="https://api.siliconflow.cn/v1"` (Consider using a referral link for rewards:  [Siliconflow Referral](https://cloud.siliconflow.cn/i/WNLYbBpi))
*   `PRIMARY_MODEL=Qwen/Qwen3-14B`
*   `VL_MODEL=Pro/Qwen/Qwen2.5-VL-7B-Instruct`

### 🚀 Run Wiseflow

```bash
cd wiseflow
uv venv # Only needed the first time
source .venv/bin/activate  # Linux/macOS
# Or on Windows:
# .venv\Scripts\activate
uv sync # Only needed the first time
python -m playwright install --with-deps chromium # Only needed the first time
chmod +x run.sh # Only needed the first time
./run.sh
```

For detailed usage instructions, see [docs/manual/manual.md](./docs/manual/manual.md).

---

## 📚 Using Data in Your Applications

Wiseflow stores data in PocketBase, allowing easy access via PocketBase SDKs for Go, Javascript, Python, and other languages.

Share your projects at:

*   https://github.com/TeamWiseFlow/wiseflow_plus

---

## 🛡️ License

Licensed under [Apache2.0](LICENSE).

For commercial partnerships, contact:  **Email: zm.zhao@foxmail.com**

---

## 📬 Contact

For questions or suggestions, please open an [issue](https://github.com/TeamWiseFlow/wiseflow/issues).

---

## 🤝 Based On

Wiseflow leverages several excellent open-source projects:

*   Crawl4ai (Open-source LLM Friendly Web Crawler & Scraper) https://github.com/unclecode/crawl4ai
*   MediaCrawler (xhs/dy/wb/ks/bilibili/zhihu crawler) https://github.com/NanmiCoder/MediaCrawler
*   NoDriver (Providing a blazing fast framework for web automation, webscraping, bots and any other creative ideas...) https://github.com/ultrafunkamsterdam/nodriver
*   Pocketbase (Open Source realtime backend in 1 file) https://github.com/pocketbase/pocketbase
*   Feedparser (Parse feeds in Python) https://github.com/kurtmckee/feedparser
*   SearXNG (a free internet metasearch engine which aggregates results from various search services and databases) https://github.com/searxng/searxng

---

## Citation

If you use or reference this project:

```
Author：Wiseflow Team
https://github.com/TeamWiseFlow/wiseflow
Licensed under Apache2.0
```

---

## 友情链接

[<img src="docs/logos/SiliconFlow.png" alt="siliconflow" width="360">](https://siliconflow.com/)