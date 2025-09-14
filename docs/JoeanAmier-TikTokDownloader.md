<div align="center">
<img src="./static/images/DouK-Downloader.png" alt="DouK-Downloader" height="256" width="256"><br>
<h1>DouK-Downloader</h1>
<p>简体中文 | <a href="README_EN.md">English</a></p>
</div>

[![Trendshift](https://trendshift.io/api/badge/repositories/6222)](https://trendshift.io/repositories/6222)
<br>
[![GitHub License](https://img.shields.io/github/license/JoeanAmier/TikTokDownloader?style=flat-square)](https://github.com/JoeanAmier/TikTokDownloader/blob/master/LICENSE)
[![GitHub Forks](https://img.shields.io/github/forks/JoeanAmier/TikTokDownloader?style=flat-square&color=55efc4)](https://github.com/JoeanAmier/TikTokDownloader/network/members)
[![GitHub Stars](https://img.shields.io/github/stars/JoeanAmier/TikTokDownloader?style=flat-square&color=fda7df)](https://github.com/JoeanAmier/TikTokDownloader/stargazers)
[![GitHub Code Size](https://img.shields.io/github/languages/code-size/JoeanAmier/TikTokDownloader?style=flat-square&color=a29bfe)](https://github.com/JoeanAmier/TikTokDownloader)
<br>
[![Python Version](https://img.shields.io/badge/Python-3.12-b8e994?style=flat-square&logo=python&labelColor=3dc1d3)](https://www.python.org/downloads/)
[![GitHub Release](https://img.shields.io/github/v/release/JoeanAmier/TikTokDownloader?style=flat-square&color=48dbfb)](https://github.com/JoeanAmier/TikTokDownloader/releases/latest)
[![Sourcery](https://img.shields.io/badge/Sourcery-enabled-884898?style=flat-square&color=1890ff)](https://sourcery.ai/)
[![Docker](https://img.shields.io/badge/Docker-badc58?style=flat-square&logo=docker)](https://www.docker.com/)
[![GitHub All Downloads](https://img.shields.io/github/downloads/JoeanAmier/TikTokDownloader/total?style=flat-square&color=ffdd59)](https://github.com/JoeanAmier/TikTokDownloader/releases)

## DouK-Downloader: Your One-Stop Solution for Downloading TikTok and Douyin Content

DouK-Downloader is a powerful, open-source tool that lets you effortlessly download videos, images, and data from TikTok and Douyin. Find the original repo [here](https://github.com/JoeanAmier/TikTokDownloader)!

---

## Key Features

*   ✅ **Comprehensive Content Downloads:** Download videos, images, and more from TikTok and Douyin, including posts, likes, collections, live streams, and more.
*   ✅ **High-Quality Downloads:** Get videos in the highest available resolution and original format.
*   ✅ **Account & Collection Downloads:** Batch download content from user accounts and collections.
*   ✅ **Data Collection & Analysis:** Gather detailed data from TikTok and Douyin, including comments and trending topics.
*   ✅ **Multiple Download Methods:** Supports various download methods, including link-based, account-based, and collection-based downloads.
*   ✅ **User-Friendly Interface:** Offers both terminal and web UI modes for easy use.
*   ✅ **API Access:** Provides a Web API for programmatic access to download functions.
*   ✅ **Cross-Platform Support:**  Works on Windows, macOS, and Linux.
*   ✅ **Proxy Support:** Use proxies for data collection.
*   ✅ **Docker Support:** Run the application in a Docker container.

---

## Functionality Overview

<details>
<summary>Feature List</summary>

*   ✅ Download Douyin/TikTok videos (no watermarks)
*   ✅ Download Douyin/TikTok images (no watermarks)
*   ✅ Download Douyin/TikTok live streams
*   ✅ Download Douyin/TikTok original videos
*   ✅ Download Douyin/TikTok user posts/likes/favorites
*   ✅ Scrape Douyin/TikTok account details
*   ✅ Batch download via links
*   ✅ Multi-account download
*   ✅ Skip already downloaded files
*   ✅ Data persistence (CSV/XLSX/SQLite)
*   ✅ Download dynamic/static cover images
*   ✅ Get Douyin/TikTok live stream addresses
*   ✅ Use ffmpeg for live stream downloads
*   ✅ Web UI (future development)
*   ✅ Scrape Douyin comments
*   ✅ Download Douyin/TikTok collections
*   ✅ Get statistics (likes/favorites)
*   ✅ Filter by publication date
*   ✅ Incremental downloads for accounts
*   ✅ Proxy support
*   ✅ Web API support
*   ✅ Multi-threaded downloading
*   ✅ File integrity checks
*   ✅ Custom filtering rules
*   ✅ File organization
*   ✅ Customizable file size limits
*   ✅ Resume downloads
*   ✅ Clipboard monitoring

</details>

---

## Screenshots & Usage

### Terminal Mode

<p>For configuration, it is recommended to use a configuration file. Learn more in the <a href="https://github.com/JoeanAmier/TikTokDownloader/wiki/Documentation">documentation</a>.</p>

![Terminal Screenshot 1](docs/screenshot/终端交互模式截图CN1.png)
*****
![Terminal Screenshot 2](docs/screenshot/终端交互模式截图CN2.png)
*****
![Terminal Screenshot 3](docs/screenshot/终端交互模式截图CN3.png)

### Web API Interface

![WebAPI Screenshot 1](docs/screenshot/WebAPI模式截图CN1.png)
*****
![WebAPI Screenshot 2](docs/screenshot/WebAPI模式截图CN2.png)

> Access the auto-generated API documentation by visiting `http://127.0.0.1:5555/docs` or `http://127.0.0.1:5555/redoc` after starting the API server.

### API Usage Example

```python
from httpx import post
from rich import print

def demo():
    headers = {"token": ""}
    data = {
        "detail_id": "0123456789",
        "pages": 2,
    }
    api = "http://127.0.0.1:5555/douyin/comment"
    response = post(api, json=data, headers=headers)
    print(response.json())

demo()
```

---

## Getting Started

### Quickstart

*   **For macOS and Windows 10+ users:** Download pre-built executables from the [Releases](https://github.com/JoeanAmier/TikTokDownloader/releases/latest) or [Actions](https://github.com/JoeanAmier/TikTokDownloader/actions) pages.
*   **For building executables:** Use the GitHub Actions workflow (see instructions below).

**Instructions:**

1.  **Run the executable** OR **Configure environment:**
    *   **Running Executable:**
        1.  Download the executable from [Releases](https://github.com/JoeanAmier/TikTokDownloader/releases/latest) or Actions.
        2.  Extract and run the `main` executable.
    *   **Configuring Environment:**
        1.  Install Python 3.12 from [Python](https://www.python.org/).
        2.  Download the source code or Releases.
        3.  Run `python -m venv venv` (optional) to create a virtual environment.
        4.  Activate the virtual environment with `.\venv\Scripts\activate.ps1` or `venv\Scripts\activate` (optional).
        5.  Run `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`.
        6.  Run `python .\main.py` or `python main.py` to launch DouK-Downloader.
2.  Accept the disclaimer.
3.  Configure your Cookie in the configuration file. See the "Cookie Guide" below.
4.  Use the interface, selecting `Terminal Mode` -> `Batch download from links` -> `Enter links`.
5.  Enter the Douyin/TikTok video links.
6.  For more details, refer to the [Project Documentation](https://github.com/JoeanAmier/TikTokDownloader/wiki/Documentation).

*   Recommended: Use [Windows Terminal](https://learn.microsoft.com/en-us/windows/terminal/install) (built-in in Windows 11).

---

### Docker Container

1.  **Get the Image:**
    *   Method 1: Build from the `Dockerfile`.
    *   Method 2: Pull the image using `docker pull joeanamier/tiktok-downloader`.
    *   Method 3: Pull the image using `docker pull ghcr.io/joeanamier/tiktok-downloader`.
2.  **Create the Container:**
    ```bash
    docker run --name <container_name (optional)> -p <host_port>:5555 -v tiktok_downloader_volume:/app/Volume -it <image_name>
    ```
    **Note:** Replace `<image_name>` with the name used to pull the image (e.g., `joeanamier/tiktok-downloader` or `ghcr.io/joeanamier/tiktok-downloader`).
3.  **Run the Container:**
    *   Start: `docker start -i <container_name/container_ID>`
    *   Restart: `docker restart -i <container_name/container_ID>`
4. Docker containers cannot access the host file system, some features are unavailable, such as: Read Cookie from the browser. Report any other feature errors!

---

## Cookie Guide

*   Refer to the [Cookie Extraction Guide](https://github.com/JoeanAmier/TikTokDownloader/blob/master/docs/Cookie%E8%8E%B7%E5%8F%96%E6%95%99%E7%A8%8B.md) for instructions.
    *   Cookies only need to be updated when they expire, not every time you run the program.
    *   Cookie quality impacts the resolution of downloaded videos. Update if the highest resolution is unavailable.
    *   Update or use a logged-in cookie when data retrieval fails.

---

## Other Notes

*   Press Enter to go back in the menu or type `Q` or `q` to exit.
*   Downloading liked/collected posts requires all data to be fetched, potentially taking longer. Adjust `max_pages` to limit the number of requests.
*   You need login Cookie for private account and follow it for their post.
*   The file name will update with nickname or tag if the account details change.
*   Downloaded files are first stored in a temporary folder.
*   The "download collections" mode only supports the logged-in account's favorites.
*   Use the `proxy` parameter in `settings.json` to enable proxy usage.
*   Use a [JSON online tool](https://try8.cn/tool/format/json) to edit configuration files.
*   Avoid line breaks in input links.
*   This project does not support the download of paid content.
*   Run with administrator privileges on Windows to read Chromium, Chrome, and Edge browser cookies.
*   For multiple instances, copy the entire project folder.
*   Press `Ctrl + C` to stop the program/`ffmpeg` instead of closing the terminal window.

---

## Building Executables

<details>
<summary><b>Build Executables Guide (click to expand)</b></summary>

This guide shows how to build and package the latest source code of DouK-Downloader by forking the repository and using GitHub Actions!

---

### Steps

#### 1. Fork the Repository

1.  Click the **Fork** button in the top-right corner of the repository.
2.  Your fork's URL will be `https://github.com/your-username/this-repo`.

---

#### 2. Enable GitHub Actions

1.  Go to your forked repository.
2.  Go to the **Settings** tab.
3.  Go to the **Actions** tab.
4.  Select **General**.
5.  Under **Actions permissions**, choose **Allow all actions and reusable workflows**, then click **Save**.

---

#### 3. Trigger the Build Workflow

1.  In your forked repository, go to the **Actions** tab.
2.  Find the workflow named **Build Executable**.
3.  Click the **Run workflow** button:
    *   Select `master` or `develop` branch.
    *   Click **Run workflow**.

---

#### 4. Monitor the Build Progress

1.  In the **Actions** tab, view the workflow runs.
2.  Click a run to view detailed logs.

---

#### 5. Download Build Results

1.  After the build finishes, go to the workflow run page.
2.  Find the **Artifacts** section.
3.  Download and save the packaged program.

---

### Important Considerations

1.  **Resource Usage:**
    *   GitHub provides a free usage tier for Actions (2000 minutes per month for ordinary users).

2.  **Code Modification:**
    *   Customize build with code changes in your fork.
    *   Trigger builds after changes.

3.  **Syncing with the Main Repository:**
    *   Regularly sync your fork to get updates from the main repo.

---

### Actions FAQs

#### Q1: Workflow trigger issues?

A: Check that you enabled Actions.

#### Q2: Build failures?

A: Check the build logs, code for errors, and dependencies. If unresolved, file an issue.

#### Q3: Can I use the main repository's Actions directly?

A: No. Fork the repo to run builds due to permission restrictions.

</details>

---

## Updating the Program

*   **Option 1:**  Download and extract. Copy the `_internal\Volume` folder from the old version to the `_internal` folder in the new version.
*   **Option 2:** Download and extract (do not run). Copy all files to overwrite the old version.

---

## ⚠️ Disclaimer

*   Users are responsible for their use of this project, at their own risk. The author is not responsible for any losses or risks arising from its use.
*   The code is based on existing knowledge and technologies. The author strives for correctness and security but does not guarantee the absence of errors.
*   Third-party libraries, plugins, or services have their own licenses. Users must follow those agreements, and the author is not responsible for their stability or compliance.
*   Users must comply with legal regulations and must not violate copyrights. Developers do not participate in or support illegal content.
*   The author is not responsible for the legality of user data collection, storage, or transmission. Users are responsible for legal compliance.
*   Do not associate the project author with your use of this project or request their responsibility for any losses.
*   The author does not offer paid versions or commercial services.
*   Secondary development, modification, or compilation by others is unrelated to the original author. Users assume all risks and responsibilities.
*   The project grants no patent licenses. Users assume all risks and responsibilities for patent disputes.
*   The author can terminate service to any user violating the terms.
*   The author reserves the right to update this disclaimer.

**Please consider and accept the above disclaimer before using the project. If you disagree, do not use the code. Use of the project implies that you fully understand and accept the disclaimer.**

---

<h1>🌟 Contribution Guide</h1>

*   Contribute to the project! Read this guide to make your contributions successful.

*   Always start with the latest code from the `develop` branch.
*   Submit related changes in separate pull requests.
*   Make pull requests focused on one feature or fix.
*   Follow the existing code style, using Ruff for formatting.
*   Write readable, well-commented code.
*   Include a clear commit message (`<type>: <brief description>`).
*   Submit pull requests to the `develop` branch for initial review.
*   Communicate with the author.

**Resources:**

*   [Contributor Covenant](https://www.contributor-covenant.org/zh-cn/version/2/1/code_of_conduct/)
*   [How to contribute to Open Source](https://opensource.guide/zh-hans/how-to-contribute/)

---

## ♥️ Support the Project

If DouK-Downloader is helpful, consider giving it a **Star** ⭐!

<table>
<thead>
<tr>
<th align="center">WeChat</th>
<th align="center">Alipay</th>
</tr>
</thead>
<tbody><tr>
<td align="center"><img src="./docs/微信赞助二维码.png" alt="WeChat QR Code" height="200" width="200"></td>
<td align="center"><img src="./docs/支付宝赞助二维码.png" alt="Alipay QR Code" height="200" width="200"></td>
</tr>
</tbody>
</table>
<p>Donate to DouK-Downloader if you wish.</p>

---

## 💰 Project Sponsorship

### DartNode

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")

### ZMTO

<a href="https://www.zmto.com/"><img src="https://console.zmto.com/templates/2019/dist/images/logo_dark.svg" alt="ZMTO"></a>
<p><a href="https://www.zmto.com/">ZMTO</a> is a cloud infrastructure provider offering robust technology and expert support.</p>

### TikHub

<p><a href="https://tikhub.io/">TikHub</a> provides third-party API services.</p>
<p>Get free usage by checking in daily, or use my referral link: <a href="https://user.tikhub.io/users/signup?referral_code=ZrdH8McC">https://user.tikhub.io/users/signup?referral_code=ZrdH8McC</a> or referral code `ZrdH8McC` for $2 in credit!</p>

---

## ✉️ Contact

*   Email: yonglelolu@foxmail.com
*   WeChat: Downloader_Tools
*   WeChat Public Account: Downloader Tools
*   <b>Discord Community</b>: <a href="https://discord.com/invite/ZYtmgKud9Y">Join Discord</a>
*   QQ Group (Project Discussions): <a href="https://github.com/JoeanAmier/TikTokDownloader/blob/master/docs/QQ%E7%BE%A4%E8%81%8A%E4%BA%8C%E7%BB%B4%E7%A0%81.png">Join QQ Group</a>

<p>✨ <b>Other Projects:</b></p>

*   **XHS-Downloader (小红书, XiaoHongShu, RedNote):**  <a href="https://github.com/JoeanAmier/XHS-Downloader">https://github.com/JoeanAmier/XHS-Downloader</a>
*   **KS-Downloader (快手, KuaiShou):**  <a href="https://github.com/JoeanAmier/KS-Downloader">https://github.com/JoeanAmier/KS-Downloader</a>

---

<h1>⭐ Star History</h1>

<p>
<img alt="Star History Chart" src="https://api.star-history.com/svg?repos=JoeanAmier/TikTokDownloader&amp;type=Timeline"/>
</p>

---

## 💡 References

*   [Johnserf-Seed/f2](https://github.com/Johnserf-Seed/f2)
*   [Johnserf-Seed/TikTokDownload](https://github.com/Johnserf-Seed/TikTokDownload)
*   [Evil0ctal/Douyin\_TikTok\_Download\_API](https://github.com/Evil0ctal/Douyin_TikTok_Download_API)
*   [NearHuiwen/TiktokDouyinCrawler](https://github.com/NearHuiwen/TiktokDouyinCrawler)
*   [ihmily/DouyinLiveRecorder](https://github.com/ihmily/DouyinLiveRecorder)
*   [encode/httpx/](https://github.com/encode/httpx/)
*   [Textualize/rich](https://github.com/Textualize/rich)
*   [omnilib/aiosqlite](https://github.com/omnilib/aiosqlite)
*   [Tinche/aiofiles](https://github.com/Tinche/aiofiles)
*   [thewh1teagle/rookie](https://github.com/thewh1teagle/rookie)
*   [pyinstaller/pyinstaller](https://github.com/pyinstaller/pyinstaller)
*   [foss.heptapod.net/openpyxl/openpyxl](https://foss.heptapod.net/openpyxl/openpyxl)
*   [carpedm20/emoji/](https://github.com/carpedm20/emoji/)
*   [lxml/lxml](https://github.com/lxml/lxml)
*   [ffmpeg.org/ffmpeg-all.html](https://ffmpeg.org/ffmpeg-all.html)
*   [tikwm.com](https://www.tikwm.com/)