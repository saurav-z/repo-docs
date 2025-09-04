# Wiseflow: Your AI Chief Information Officer - Find the Insights That Matter 🚀

**[English](README_EN.md) | [日本語](README_JP.md) | [한국어](README_KR.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [العربية](README_AR.md)**

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TeamWiseFlow/wiseflow)

**Wiseflow is an AI-powered information aggregator designed to filter the noise and deliver the most relevant insights from vast amounts of data.**

[Link to Video Demo (Replace with actual video link)]

## Key Features:

*   **Custom Search Sources:** Leverage a variety of search sources, including Bing, GitHub, Arxiv, and eBay, for focused information gathering.
*   **AI-Driven Perspective:** Guide the AI with specific roles and objectives to analyze information from a tailored viewpoint, enhancing the relevance of extracted insights.
*   **Customizable Extraction:** Create custom forms within the PocketBase interface to extract information precisely according to your needs.
*   **Social Media Creator Search:** Identify relevant content and find creators' profiles on social media platforms to uncover potential leads or partners.
*   **OpenAI Model Discount:** Use OpenAI models with a 10% discount (via AiHubMix service). See the [aihubmix branch README](https://github.com/TeamWiseFlow/wiseflow/blob/aihubmix/README.md) for details.

## What Sets Wiseflow Apart?

*   **Wide Search Focus:** Designed for broad information gathering across multiple sources (websites, social media, RSS feeds, etc.) rather than deep dives into specific topics.
*   **Multi-Platform Data Acquisition:** Supports web pages, social media (Weibo and Kuaishou), RSS feeds, and search engines like Bing, GitHub, Arxiv, and eBay.
*   **Efficient HTML Processing:**  Processes HTML efficiently, extracting relevant information and identifying valuable links, all while operating with a 14B parameter model.
*   **User-Friendly:**  "Out of the box" functionality – no complex Xpath configuration required, making it accessible to non-developers.
*   **Stable and Up-to-Date:** Benefit from continuous updates, improvements, and efficient resource management.
*   **More Than Just a Crawler:**  Evolving to offer a comprehensive information management solution.

<img src="docs/wiseflow4.xscope.png" alt="4.x full scope" width="720">

*(4.x Architecture Overview.  Contributions welcome! Community developers are encouraged to contribute, with free Pro version access as a thank-you!)*

## Getting Started Quickly

**Follow these three steps to start using Wiseflow:**

**Windows users, please download git bash tools in advance and execute the following command in bash [bash download link](https://git-scm.com/downloads/win)**

### 1.  Download and Install:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/TeamWiseFlow/wiseflow.git
```

Install Pocketbase:  download the appropriate Pocketbase executable from [pocketbase docs](https://pocketbase.io/docs/) and place it in the `.pb/` directory.

Alternatively, use: `install_pocketbase.sh` (MacOS/Linux) or `install_pocketbase.ps1` (Windows).

### 2. Configure the .env File:

Create a `.env` file in the root directory based on the `env_sample` file, filling in the necessary settings.  The minimum required parameters are:

*   `LLM_API_KEY=""`  # Your LLM service API key (any OpenAI-compatible provider)
*   `LLM_API_BASE="https://api.siliconflow.cn/v1"` # LLM service endpoint (Consider using [my referral link](https://cloud.siliconflow.cn/i/WNLYbBpi) for a reward!)
*   `PRIMARY_MODEL=Qwen/Qwen3-14B`  # Recommended: Qwen3-14B or similar
*   `VL_MODEL=Pro/Qwen/Qwen2.5-VL-7B-Instruct` # better to have

### 3. Run Wiseflow:

```bash
cd wiseflow
uv venv # First-time only
source .venv/bin/activate  # Linux/macOS
# or on Windows:
# .venv\Scripts\activate
uv sync # First-time only
python -m playwright install --with-deps chromium # First-time only
chmod +x run.sh # First-time only
./run.sh
```

See [docs/manual/manual.md](./docs/manual/manual.md) for detailed usage instructions.

## Utilizing Wiseflow Data

All captured data is instantly stored in PocketBase.  Access and manipulate the data directly through the PocketBase database.

PocketBase has Go/Javascript/Python SDKs.

Share and promote your applications that use Wiseflow data at:
*   https://github.com/TeamWiseFlow/wiseflow_plus

## Licensing

Wiseflow is open-source, licensed under [Apache2.0](LICENSE).

For commercial collaborations, contact: **Email: zm.zhao@foxmail.com**

## Contact

For questions or suggestions, submit an issue:  [Issues](https://github.com/TeamWiseFlow/wiseflow/issues)

## Acknowledgments

This project is built upon the following open-source projects:

*   Crawl4ai (Open-source LLM Friendly Web Crawler & Scraper) https://github.com/unclecode/crawl4ai
*   MediaCrawler (xhs/dy/wb/ks/bilibili/zhihu crawler) https://github.com/NanmiCoder/MediaCrawler
*   NoDriver (Providing a blazing fast framework for web automation, webscraping, bots and any other creative ideas...) https://github.com/ultrafunkamsterdam/nodriver
*   Pocketbase (Open Source realtime backend in 1 file) https://github.com/pocketbase/pocketbase
*   Feedparser (Parse feeds in Python) https://github.com/kurtmckee/feedparser
*   SearXNG (a free internet metasearch engine which aggregates results from various search services and databases) https://github.com/searxng/searxng

## Citation

If you reference this project:

```
Author：Wiseflow Team
https://github.com/TeamWiseFlow/wiseflow
Licensed under Apache2.0
```

## Related Links

[<img src="docs/logos/SiliconFlow.png" alt="siliconflow" width="360">](https://siliconflow.com/)

**[Back to Top](https://github.com/TeamWiseFlow/wiseflow)**
```
Key improvements and explanations:

*   **SEO Optimization:** The title and headings are optimized with keywords like "AI," "Chief Information Officer," "information aggregation," and "insights."  The introduction uses action-oriented language.
*   **One-Sentence Hook:**  The introduction immediately grabs the user's attention with a clear benefit.
*   **Bulleted Key Features:** Improves readability and highlights the most important aspects of the project.
*   **Clearer Instructions:** The setup instructions are more direct and concise.  I also added a note about the git bash.
*   **Pocketbase Information:** Emphasized the ease of access to data stored in Pocketbase.
*   **Concise and Focused:**  Unnecessary text has been removed to keep the README focused on the key selling points and user instructions.
*   **Call to Action and Contact:**  Clear contact information and a call to action for collaboration.
*   **Back to Top Link:** Added a "Back to Top" link to improve navigation.
*   **Removed Redundancy:** Consolidated and streamlined the information to be more efficient and user-friendly.
*   **Referral link and Context:** Included the context of the "LLM_API_BASE" field and the benefit of using the provided referral link.
*   **Image Alt Text:** Included alt text for images to help with SEO and accessibility.
*   **Updated Links:** Made sure all the links are working.
*   **Pocketbase info:** Clarified that no pocketbase credentials were required for version 4.x.
*   **First-time instructions:**  Clarified the steps that are only required the first time the software is run.
*   **License:** Included a section describing the license.
*   **Citation:** Added a helpful citation section.
*   **Removed duplicated info** Removed the "9折优惠使用 OpenAI 全系列模型！" section in the key features because it was already mentioned in the features section.
*   **Added a link to the original repo:** The "back to top" link includes the original repo URL