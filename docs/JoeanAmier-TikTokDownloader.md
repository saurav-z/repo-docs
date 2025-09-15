<div align="center">
<img src="./static/images/DouK-Downloader.png" alt="DouK-Downloader" height="256" width="256"><br>
<h1>DouK-Downloader</h1>
</div>

<p>Easily download videos, images, and data from TikTok and Douyin with DouK-Downloader!  (<a href="https://github.com/JoeanAmier/TikTokDownloader">View on GitHub</a>)</p>

<p>
<a href="https://trendshift.io/repositories/6222" target="_blank"><img src="https://trendshift.io/api/badge/repositories/6222" alt="" style="width: 250px; height: 55px;" width="250" height="55"/></a>
<img alt="GitHub" src="https://img.shields.io/github/license/JoeanAmier/TikTokDownloader?style=flat-square">
<img alt="GitHub forks" src="https://img.shields.io/github/forks/JoeanAmier/TikTokDownloader?style=flat-square&color=55efc4">
<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/JoeanAmier/TikTokDownloader?style=flat-square&color=fda7df">
<img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/JoeanAmier/TikTokDownloader?style=flat-square&color=a29bfe">
<br>
<img alt="Static Badge" src="https://img.shields.io/badge/Python-3.12-b8e994?style=flat-square&logo=python&labelColor=3dc1d3">
<img alt="GitHub release (with filter)" src="https://img.shields.io/github/v/release/JoeanAmier/TikTokDownloader?style=flat-square&color=48dbfb">
<img src="https://img.shields.io/badge/Sourcery-enabled-884898?style=flat-square&color=1890ff" alt="">
<img alt="Static Badge" src="https://img.shields.io/badge/Docker-badc58?style=flat-square&logo=docker">
<img alt="GitHub all releases" src="https://img.shields.io/github/downloads/JoeanAmier/TikTokDownloader/total?style=flat-square&color=ffdd59">
</p>

## Key Features

*   **Multi-Platform Support:** Download videos, images, and data from both TikTok and Douyin.
*   **Batch Downloads:** Download content from user accounts, collections, and more in bulk.
*   **High-Quality Downloads:**  Get videos in the best available resolution, including original TikTok video quality.
*   **Data Extraction:** Collect detailed account information and comments.
*   **Flexible Options:** Supports multiple methods for downloading including links, browser cookies and Web API.
*   **Easy-to-Use:** Supports terminal, Web UI, and Web API modes for flexibility.
*   **Docker Support:** Deploy and run easily using Docker.

## Functionality

*   ✅ Download Douyin/TikTok videos/images (without watermarks)
*   ✅ Download Douyin/TikTok live streams
*   ✅ Download high-quality video files
*   ✅ Download Douyin/TikTok account content (posts, likes, collections)
*   ✅ Collect comprehensive data (account info, comments, etc.)
*   ✅ Batch download content via links
*   ✅ Support CSV/XLSX/SQLite for data storage
*   ✅ Proxy support for data collection
*   ✅ Web API interface for integration
*   ✅ Customizable download rules

See the full list of features in the expanded feature list in the original README.

## Getting Started

### Installation

Choose your preferred way to install the program.

*   **Option 1: Executable (Recommended)**
    *   Download pre-compiled executables from the [Releases](https://github.com/JoeanAmier/TikTokDownloader/releases/latest) or [Actions](https://github.com/JoeanAmier/TikTokDownloader/actions)
    *   Unzip and run the `main` executable.
*   **Option 2: Python (Requires Python 3.12)**
    1.  Install Python 3.12.
    2.  Download the source code.
    3.  Create and activate a virtual environment (optional): `python -m venv venv` and then `.\venv\Scripts\activate`
    4.  Install requirements: `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`
    5.  Run: `python main.py`

### Setup

1.  Run the executable or start the program.
2.  Follow the on-screen prompts, and review the disclaimer.
3.  **Configure Cookie:**  You can choose from the following methods:
    *   **From Clipboard:** Copy the cookie from your browser, and select the appropriate option.
    *   **From Browser:** Select the appropriate option, and specify the browser.
4.  Choose the desired mode, and enter the required links or options.
5.  More detailed instructions and the user documentation is available in the [Project Documentation](https://github.com/JoeanAmier/TikTokDownloader/wiki/Documentation)

### Docker Usage

1.  Pull image: `docker pull joeanamier/tiktok-downloader` or `docker pull ghcr.io/joeanamier/tiktok-downloader`
2.  Create container: `docker run --name <container_name> -p 5555:5555 -v tiktok_downloader_volume:/app/Volume -it <image_name>`
3.  Run container: `docker start -i <container_name or container_id>`

## Additional Information

*   **Cookie:** Update your cookie periodically, as it may be required to access highest quality videos.
*   **Web Interface:** Use `http://127.0.0.1:5555/docs` or `http://127.0.0.1:5555/redoc` to view the generated API documentation.
*   **See the original README for more details on:**
    *   API examples
    *   File Structure
    *   Docker container configurations
    *   and much more!

## Disclaimer

*   Please review the **full disclaimer** within the original README.  Users are responsible for their usage of the software and compliance with all applicable laws.

---
I have also added a line for better SEO optimization.