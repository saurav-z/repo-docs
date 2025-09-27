# Wiseflow: Your AI-Powered Chief Intelligence Officer

**[English](README_EN.md) | [日本語](README_JP.md) | [한국어](README_KR.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [العربية](README_AR.md)**

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TeamWiseFlow/wiseflow)

**Wiseflow empowers you to extract valuable insights from the vast ocean of information, leveraging large language models (LLMs) to filter noise and surface critical information from diverse sources.**

[View the original repository](https://github.com/TeamWiseFlow/wiseflow)

<img src="https://github.com/user-attachments/assets/48998353-6c6c-4f8f-acae-dc5c45e2e0e6" alt="Wiseflow Demo">

## Key Features of Wiseflow

*   **Wide Search Capabilities:** Efficiently gathers information from a broad range of sources, including web pages, social media (Weibo, Kuaishou), RSS feeds, and search engines (Bing, GitHub, Arxiv).
*   **AI-Driven Filtering:** Utilizes LLMs to identify and extract relevant information, filtering out noise and focusing on your specific interests.
*   **Customizable Search:** Configure search sources and set roles and objectives for the LLM to guide analysis and extraction.
*   **Advanced Web Scraping:**  The latest version utilizes your local Chrome browser to enhance web scraping capabilities, improving reliability and enabling login-based access to websites.
*   **Custom Extraction Templates:** Create custom forms within the PocketBase interface for precise data extraction, making it easier to find potential clients, partners, or investors.
*   **Social Media Discovery:** Identify creators and content on social media platforms, enabling lead generation and market research.

## What Makes Wiseflow Unique?

*   **Broad Source Coverage:** Access data from a wide array of online sources.
*   **Intelligent HTML Processing:** LLMs automatically extract information and identify relevant links.
*   **LLM-Integrated Crawling:** Reduces the risk of platform detection by integrating LLMs during the crawling process.
*   **User-Friendly Interface:** Designed for ease of use, eliminating the need for manual XPaths.
*   **Continuous Improvement:** Benefit from ongoing updates and improvements, ensuring stability and efficiency.

## 🚀 Getting Started

Follow these simple steps to get started with Wiseflow!

**Important:**  Starting with version 4.2, you *must* have Google Chrome installed (using the default installation path).

### 1.  Clone the Repository & Install Dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/TeamWiseFlow/wiseflow.git
```

After cloning, you'll need to install the PocketBase application. Download the appropriate version for your operating system from the [PocketBase docs](https://pocketbase.io/docs/) and place it in the `.pb` directory.  Alternatively, you can try using the provided `install_pocketbase.sh` (MacOS/Linux) or `install_pocketbase.ps1` (Windows) scripts.

### 2.  Configure the Environment

Create a `.env` file in the project's root directory, using `env_sample` as a guide.  At a minimum, you'll need to configure:

*   `LLM_API_KEY`:  Your API key for the LLM service.
*   `LLM_API_BASE`: The base URL for your LLM API. (Siliconflow is recommended)
*   `PRIMARY_MODEL`: The primary LLM model to use.
*   `VL_MODEL`: An optional visual analysis model.

### 3.  Run Wiseflow

```bash
cd wiseflow
uv venv # First time only
source .venv/bin/activate  # Linux/macOS
# Or on Windows:
# .venv\Scripts\activate
uv sync # First time only
chmod +x run.sh # First time only
./run.sh
```

For detailed usage instructions, please consult the [manual](./docs/manual/manual.md).

## Using Wiseflow Data

All extracted data is stored in PocketBase. You can access and utilize the data by interacting with the PocketBase database using its Go/Javascript/Python SDKs.

## License & Support

*   **License:**  See the [LICENSE](LICENSE) file for the updated open-source license (as of v4.2).
*   **Commercial Collaboration:** Contact zm.zhao@foxmail.com for commercial inquiries.
*   **Contact:**  Share your feedback and suggestions through [issues](https://github.com/TeamWiseFlow/wiseflow/issues).

## Acknowledgements

This project is built upon the following open-source projects:  (List of dependencies with links)

## Citation

```
Author：Wiseflow Team
https://github.com/TeamWiseFlow/wiseflow
```

## External Links

*   [SiliconFlow](https://siliconflow.com/)