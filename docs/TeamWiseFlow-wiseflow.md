# Wiseflow: Your AI Chief Intelligence Officer 🧠

**Unlock valuable insights from the ocean of information with Wiseflow, the AI-powered tool that sifts through the noise to deliver the information you truly need.**  Find the original repository on [GitHub](https://github.com/TeamWiseFlow/wiseflow).

**[English](README_EN.md) | [日本語](README_JP.md) | [한국어](README_KR.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [العربية](README_AR.md)**

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TeamWiseFlow/wiseflow)

## Key Features

*   **Wide Search Focus:** Designed for broad information gathering across multiple sources, perfect for industry intelligence, background checks, and lead generation.
*   **Customizable Search Sources:**  Choose from a variety of sources including Bing, GitHub, arXiv, and eBay, using native APIs.
*   **AI-Driven Analysis:**  Define roles and objectives for AI analysis to gain specific perspectives on your data.
*   **Custom Extraction:**  Create custom forms within the PocketBase interface to extract specific information from web pages.
*   **Social Media Search:**  Find content and creators on social media platforms.
*   **Platform Agnostic:** Capable of retrieving information from websites, social media (e.g., Weibo, Kuaishou), RSS feeds, and more.
*   **Efficient Architecture:** Uses an optimized HTML processing flow and a 14B parameter model for efficient information extraction.
*   **User-Friendly:**  Ready to use, eliminating the need for manual Xpath configuration.
*   **Continuous Improvement:**  Regular updates ensure stability, availability, and efficient resource usage.

<img src="docs/wiseflow4.xscope.png" alt="4.x full scope" width="720">

_(4.x Architecture Overview.  Areas within the dotted line are under development.  Community contributions are welcome!)_

## Wiseflow 4.1: What's New?

*   **Custom Search Sources:** Configure search sources for focused information gathering.
*   **Role-Based AI Analysis:** Guide LLMs to analyze information from a specific perspective.
*   **Customizable Extraction Mode:** Create custom forms for precise information extraction.
*   **Creator Discovery:** Locate content creators and their information on social platforms.

**For more details on version 4.1, check the [CHANGELOG](CHANGELOG.md)**

## Discounted OpenAI Models! 💰

Use OpenAI models at a 10% discount within the Wiseflow application via the AiHubMix interface.  To access the discount, switch to the `aihubmix` branch.  See the [README](https://github.com/TeamWiseFlow/wiseflow/blob/aihubmix/README.md) for details.

## Getting Started Quickly

**Follow these three steps to begin using Wiseflow:**

**Windows users should install the git bash tool first.  Follow the link to download [git bash](https://git-scm.com/downloads/win) then run the following commands in the bash terminal:**

### 1.  Clone the Repository & Install Dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/TeamWiseFlow/wiseflow.git
```

This installs `uv`.

### 2.  Install PocketBase

Download PocketBase from the [PocketBase docs](https://pocketbase.io/docs/) and place it in the  `.pb/` directory of your project.

Alternatively, use `install_pocketbase.sh` (macOS/Linux) or `install_pocketbase.ps1` (Windows).

### 3. Configure the .env File

Create a `.env` file in the root directory of the project based on the `env_sample` file.  Fill in the necessary settings.  Minimum parameters required for 4.x:

```
LLM_API_KEY=""  # LLM service key (any OpenAI-compatible provider; omit if using a local Ollama deployment)
LLM_API_BASE="https://api.siliconflow.cn/v1" # LLM API endpoint (recommended: SiliconFlow; use my [referral link](https://cloud.siliconflow.cn/i/WNLYbBpi) for a bonus!)
PRIMARY_MODEL=Qwen/Qwen3-14B  # Recommended: Qwen3-14B or similar
VL_MODEL=Pro/Qwen/Qwen2.5-VL-7B-Instruct  # Recommended
```

### 4. Run Wiseflow

```bash
cd wiseflow
uv venv  #  Run only the first time
source .venv/bin/activate  # Linux/macOS
# OR on Windows:
# .venv\Scripts\activate
uv sync  # Run only the first time
python -m playwright install --with-deps chromium  # Run only the first time
chmod +x run.sh # Run only the first time
./run.sh
```

Refer to [docs/manual/manual.md](./docs/manual/manual.md) for detailed usage instructions.

## Integrating with Your Applications

Wiseflow stores all extracted data in PocketBase. You can directly access this data by interacting with the PocketBase database using its SDKs (Go/Javascript/Python).

Share your applications and development use cases in the  [wiseflow\_plus](https://github.com/TeamWiseFlow/wiseflow_plus)  repository.

## License

This project is open-source, licensed under [Apache2.0](LICENSE).

For commercial inquiries, contact **Email：zm.zhao@foxmail.com**

## Contact

For questions or suggestions, please use the [issue tracker](https://github.com/TeamWiseFlow/wiseflow/issues).

## Acknowledgements

This project leverages the following open-source projects:

*   Crawl4ai (Open-source LLM Friendly Web Crawler & Scraper) https://github.com/unclecode/crawl4ai
*   MediaCrawler (xhs/dy/wb/ks/bilibili/zhihu crawler) https://github.com/NanmiCoder/MediaCrawler
*   NoDriver (Providing a blazing fast framework for web automation, webscraping, bots and any other creative ideas...) https://github.com/ultrafunkamsterdam/nodriver
*   Pocketbase (Open Source realtime backend in 1 file) https://github.com/pocketbase/pocketbase
*   Feedparser (Parse feeds in Python) https://github.com/kurtmckee/feedparser
*   SearXNG (a free internet metasearch engine which aggregates results from various search services and databases) https://github.com/searxng/searxng

## Citation

Please cite this project if you use it in your work:

```
Author：Wiseflow Team
https://github.com/TeamWiseFlow/wiseflow
Licensed under Apache2.0
```

## Related Links

[<img src="docs/logos/SiliconFlow.png" alt="siliconflow" width="360">](https://siliconflow.com/)