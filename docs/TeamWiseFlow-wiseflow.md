# Wiseflow: Your AI-Powered Chief Intelligence Officer

**Uncover crucial insights from vast information sources using large language models.** ([Original Repository](https://github.com/TeamWiseFlow/wiseflow))

Wiseflow empowers you to filter the noise and extract valuable insights from massive datasets, helping you stay informed and ahead.

**Key Features:**

*   **Web Scraping & Content Aggregation:** Gather information from websites, social media (Weibo, Kuaishou), RSS feeds, and search engines (Bing, GitHub, Arxiv).
*   **AI-Driven Information Extraction:** Leverage large language models to analyze and extract relevant information based on your focus points, even from local Chrome browsers.
*   **Customizable Search Sources:** Refine your information gathering with tailored search sources like Bing, GitHub, and Arxiv.
*   **Role-Based Analysis:** Guide LLMs to analyze information from a specific perspective or with a defined purpose for more tailored results.
*   **Custom Extraction Forms:** Create and apply custom forms for precise data extraction and structured results.
*   **Social Media Creator Search:** Locate content creators and their information on social media platforms based on your focus points.
*   **LLM Model Recommendations:** Get guidance on the best LLM models for optimal performance and cost efficiency.
*   **Wide Search Approach:** Efficiently collect a broad range of information for tasks like industry intelligence, background checks, and lead generation, designed specifically for "wide search" scenarios.
*   **PocketBase Integration:** All scraped data is instantly stored in PocketBase, allowing easy access and integration into your own applications.

## What's New in Version 4.2?

Wiseflow 4.2 significantly enhances web scraping capabilities by leveraging your local Chrome browser, increasing reliability and enabling features like persistent user data and script support.  Refer to the [CHANGELOG](CHANGELOG.md) for a complete list of changes.

## Get Started Quickly

**Follow these three steps to begin using Wiseflow!**

**Important:**  From version 4.2 onwards, you must have Google Chrome installed with the default installation path.

**Windows users:** Download the git bash tool.  [bash download link](https://git-scm.com/downloads/win)

### 1. Clone the Repository & Install Dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/TeamWiseFlow/wiseflow.git
```

Follow the instructions to install the Pocketbase program for your system from [pocketbase docs](https://pocketbase.io/docs/) to the [.pb](./pb/) folder.

or use install_pocketbase.sh (for MacOS/Linux) or install_pocketbase.ps1 (for Windows).

### 2. Configure Your Environment

Create a `.env` file in the root directory of the project by referencing the `env_sample` file, and fill in the necessary information.
*   LLM_API_KEY=""
*   LLM_API_BASE="https://api.siliconflow.cn/v1"
*   PRIMARY_MODEL=ByteDance-Seed/Seed-OSS-36B-Instruct
*   VL_MODEL=Pro/Qwen/Qwen2.5-VL-7B-Instruct

### 3. Run Wiseflow

```bash
cd wiseflow
uv venv # Only needed the first time
source .venv/bin/activate  # Linux/macOS
# Or on Windows:
# .venv\Scripts\activate
uv sync # Only needed the first time
chmod +x run.sh # Only needed the first time
./run.sh
```

For detailed usage instructions, see [docs/manual/manual.md](./docs/manual/manual.md)

## Integrate with Your Applications

Wiseflow's data is stored in PocketBase. You can use PocketBase's SDKs (Go/Javascript/Python) to interact with the database and access the scraped data.

Contribute and share your developments at: [wiseflow\_plus](https://github.com/TeamWiseFlow/wiseflow_plus)

## License

The project is licensed under a new open-source license from version 4.2, please see [LICENSE](LICENSE).  For commercial collaborations, contact **Email：zm.zhao@foxmail.com**.

## Contact

For any questions or suggestions, please open an [issue](https://github.com/TeamWiseFlow/wiseflow/issues).

## Acknowledgements

Wiseflow is built upon several excellent open-source projects.  (See original README for full list)

## Citation

If you use or reference Wiseflow, please cite it as follows:

```
Author：Wiseflow Team
https://github.com/TeamWiseFlow/wiseflow
```

## Related Links

[<img src="docs/logos/SiliconFlow.png" alt="siliconflow" width="360">](https://siliconflow.com/)