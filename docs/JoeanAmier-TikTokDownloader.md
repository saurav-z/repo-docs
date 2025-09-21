<div align="center">
<img src="./static/images/DouK-Downloader.png" alt="DouK-Downloader" height="256" width="256"><br>
<h1>DouK-Downloader: Download TikTok and Douyin Videos with Ease</h1>
<p>  Effortlessly download videos, images, and data from TikTok and Douyin with this powerful and versatile open-source tool. </p>
<p>简体中文 | <a href="README_EN.md">English</a></p>
<a href="https://trendshift.io/repositories/6222" target="_blank"><img src="https://trendshift.io/api/badge/repositories/6222" alt="" style="width: 250px; height: 55px;" width="250" height="55"/></a>
<br>
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
</div>
<br>

## Key Features

*   ✅ **Download Videos and Images:**  Download videos and image sets from TikTok and Douyin, including original and watermarked versions.
*   ✅ **Account & Collection Download:** Batch download videos from user accounts, likes, and collections/favorites.
*   ✅ **Live Stream Support:**  Get live stream URLs and download live videos from both TikTok and Douyin.
*   ✅ **Data Scraping:**  Collect detailed user data, comments, and trending information.
*   ✅ **Multiple Download Modes:** Offers terminal, Web UI (future development), and API access.
*   ✅ **Flexible Saving & Filtering:**  Save data in CSV, XLSX, or SQLite formats, and filter videos by upload date.
*   ✅ **Proxy & Automation:** Supports proxy usage, clipboard monitoring, and automated actions.
*   ✅ **Cross-Platform & Docker:** Available as executables for various OS and Docker containers.

<hr>

##  Project Functionality

<details>
<summary>Function List</summary>
<ul>
<li>✅ Download Douyin videos/image sets without watermarks</li>
<li>✅ Download Douyin live/animated images</li>
<li>✅ Download high-resolution video files</li>
<li>✅ Download TikTok original video quality</li>
<li>✅ Download TikTok videos/image sets without watermarks</li>
<li>✅ Download Douyin account posts/likes/favorites/collections</li>
<li>✅ Download TikTok account posts/likes</li>
<li>✅ Collect detailed Douyin/TikTok data</li>
<li>✅ Batch download video links</li>
<li>✅ Download videos with multiple accounts</li>
<li>✅ Auto-skip downloaded files</li>
<li>✅ Persistently save collected data</li>
<li>✅ Support CSV/XLSX/SQLite format data saving</li>
<li>✅ Download dynamic/static cover images</li>
<li>✅ Get Douyin live streaming addresses</li>
<li>✅ Get TikTok live streaming addresses</li>
<li>✅ Call ffmpeg to download live streaming</li>
<li>✅ Web UI interaction interface</li>
<li>✅ Collect Douyin video comment data</li>
<li>✅ Download Douyin collection videos</li>
<li>✅ Download TikTok collection videos</li>
<li>✅ Record like/favorite statistics</li>
<li>✅ Filter video upload time</li>
<li>✅ Support for incremental download of account videos</li>
<li>✅ Support for using proxies to collect data</li>
<li>✅ Support LAN remote access</li>
<li>✅ Collect Douyin account details</li>
<li>✅ Video statistics data update</li>
<li>✅ Support custom account/collection identifier</li>
<li>✅ Auto-update account nickname/identifier</li>
<li>✅ Deploy to private server</li>
<li>✅ Deploy to public server</li>
<li>✅ Collect Douyin search data</li>
<li>✅ Collect Douyin trending data</li>
<li>✅ Record downloaded video IDs</li>
<li>☑️ <del>Scan login to get Cookie</del></li>
<li>✅ Read Cookie from browser</li>
<li>✅ Support Web API calls</li>
<li>✅ Support multi-threaded video download</li>
<li>✅ File integrity processing mechanism</li>
<li>✅ Custom rule filter videos</li>
<li>✅ File archiving to save video files by folder</li>
<li>✅ Custom setting file size upper limit</li>
<li>✅ Support for file breakpoint resuming download</li>
<li>✅ Listen clipboard link download video</li>
</ul>
</details>

## Screenshots

<p><a href="https://www.bilibili.com/video/BV1d7eAzTEFs/">Watch a demo on Bilibili</a>；<a href="https://youtu.be/yMU-RWl55hg">Watch a demo on YouTube</a></p>

### Terminal Interaction Mode

<p>It is recommended to use configuration files to manage accounts. For more information, please refer to the <a href="https://github.com/JoeanAmier/TikTokDownloader/wiki/Documentation">documentation</a></p>

![Terminal Mode Screenshot](docs/screenshot/终端交互模式截图CN1.png)
*****
![Terminal Mode Screenshot](docs/screenshot/终端交互模式截图CN2.png)
*****
![Terminal Mode Screenshot](docs/screenshot/终端交互模式截图CN3.png)

### Web UI Interaction Mode

> **The project code has been refactored, and the code for this mode has not been updated yet. It will be reopened after future development is completed!**

### Web API Interface Mode

![WebAPI Mode Screenshot](docs/screenshot/WebAPI模式截图CN1.png)
*****
![WebAPI Mode Screenshot](docs/screenshot/WebAPI模式截图CN2.png)

> **After starting this mode, you can access `http://127.0.0.1:5555/docs` or `http://127.0.0.1:5555/redoc` to view the automatically generated documentation!**

#### API Call Sample Code

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

<hr>

## Getting Started

### Installation

1.  **Run Executable (Recommended):** Download pre-built executables from the [Releases](https://github.com/JoeanAmier/TikTokDownloader/releases/latest) or [Actions](https://github.com/JoeanAmier/TikTokDownloader/actions) page.
2.  **Run from Source (if you know how):**

    *   Install Python 3.12 or higher.
    *   Clone or download the source code.
    *   Create a virtual environment (optional): `python -m venv venv`
    *   Activate the virtual environment: `.\venv\Scripts\activate.ps1` (Windows) or `venv/Scripts/activate` (Linux/macOS)
    *   Install dependencies: `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`
    *   Run the downloader: `python main.py`

### Usage

1.  **Run the executable** or **Configure the environment to run**
    *   **Run the executable**
    *   Download the executable compressed package from [Releases](https://github.com/JoeanAmier/TikTokDownloader/releases/latest) or Actions
    *   After decompressing, open the program folder and double-click to run `main`
    *   **Configure environment to run**
        *   Install a Python interpreter not lower than `3.12`
        *   Download the latest source code or [Releases](https://github.com/JoeanAmier/TikTokDownloader/releases/latest)
        *   Run the `python -m venv venv` command to create a virtual environment (optional)
        *   Run the `.\venv\Scripts\activate.ps1` or `venv\Scripts\activate` command to activate the virtual environment (optional)
        *   Run `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt` to install the required modules
        *   Run `python .\main.py` or `python main.py` to start DouK-Downloader
2.  Read the DouK-Downloader disclaimer and enter the content as prompted.
3.  Write Cookie information to the configuration file
    *   **Read Cookie from clipboard**
        *   Refer to the [Cookie Extraction Tutorial](https://github.com/JoeanAmier/TikTokDownloader/blob/master/docs/Cookie%E8%8E%B7%E5%8F%96%E6%95%99%E7%A8%8B.md), copy the required Cookie to the clipboard
        *   Select the `Read Cookie from the clipboard` option, and the program will automatically read the Cookie from the clipboard and write it to the configuration file
    *   **Read Cookie from browser**
        *   Select the `Read Cookie from the browser` option and enter the browser type or serial number as prompted
    *   **<del>Scan the QR code to log in to get the Cookie</del> (Invalid)**
        *   <del>Select the `Scan QR code to log in to get Cookie` option, and the program will display a login QR code image, and use the default application to open the image</del>
        *   <del>Use the Douyin APP to scan the QR code and log in to the account</del>
        *   <del>Follow the prompts to operate, the program will automatically write the Cookie to the configuration file</del>
4.  Return to the program interface and sequentially select `Terminal interaction mode` -> `Batch download link works (general)` -> `Manually enter the work links to be collected`
5.  Enter the Douyin work link to download the work file (TikTok platform requires more initial settings, see the documentation for details)
6.  For more detailed instructions, please refer to the **<a href="https://github.com/JoeanAmier/TikTokDownloader/wiki/Documentation">Project Documentation</a>**

### Docker Container

1.  Get the image
    *   Method 1: Use the `Dockerfile` file to build the image
    *   Method 2: Use the command `docker pull joeanamier/tiktok-downloader` to pull the image
    *   Method 3: Use the command `docker pull ghcr.io/joeanamier/tiktok-downloader` to pull the image
2.  Create a container: `docker run --name container name (optional) -p host port number:5555 -v tiktok_downloader_volume:/app/Volume -it <image name>`
<br>**Note:** The `<image name>` here must be consistent with the image name you used in the first step (for example, `joeanamier/tiktok-downloader` or `ghcr.io/joeanamier/tiktok-downloader`)
3.  Run the container
    *   Start the container: `docker start -i container name/container ID`
    *   Restart the container: `docker restart -i container name/container ID`
<p>The Docker container cannot directly access the file system of the host machine, and some functions are not available, such as: `Read Cookie from browser`; if there are other functional exceptions, please feedback!</p>

<hr>

## About Cookies

[Click to view the Cookie acquisition tutorial](https://github.com/JoeanAmier/TikTokDownloader/blob/master/docs/Cookie%E8%8E%B7%E5%8F%96%E6%95%99%E7%A8%8B.md)

> *   Cookies only need to be rewritten to the configuration file after they expire, and are not required to be written to the configuration file every time the program runs!
> *   Cookies will affect the resolution of the downloaded video files. If you cannot download the highest resolution video files, please try to update the Cookies!
> *   When the program fails to obtain data, you can try to update the Cookie or use the logged-in Cookie!

<hr>

## Other Notes

*   When the program prompts the user to enter, pressing Enter directly means returning to the upper menu, and entering `Q` or `q` means ending the run.
*   Since obtaining account likes and favorite work data only returns the release date of the like/favorite work, and does not return the operation date, the program needs to obtain all the like/favorite work data before filtering by date; if the number of works is large, it may take a long time; you can control the number of requests through the `max_pages` parameter.
*   Obtaining the post data of private accounts requires the Cookie after login, and the logged-in account needs to follow the private account.
*   When batch downloading account works or collection works, if the corresponding nickname or identifier changes, the program will automatically update the nickname and identifier in the downloaded work file name.
*   When the program downloads files, it will first download the files to a temporary folder, and then move them to the storage folder after the download is complete; the temporary folder will be emptied when the program ends.
*   The `Batch download favorite works mode` currently only supports downloading the favorite works of the account corresponding to the currently logged-in Cookie, and does not support multiple accounts.
*   If you want the program to use a proxy to request data, you must set the `proxy` parameter in `settings.json`, otherwise the program will not use the proxy.
*   If your computer does not have a suitable program to edit JSON files, it is recommended to use a [JSON online tool](https://try8.cn/tool/format/json) to edit the configuration file content.
*   When the program requests the user to enter content or links, please pay attention to avoiding the content or links containing newline characters, which may cause unexpected problems.
*   This project will not support the download of paid works, please do not feed back any issues about paid works download.
*   Windows system needs to run the program as an administrator to read Chromium, Chrome, Edge browser Cookie.
*   This project has not been optimized for multiple instances of the program. If you need multiple instances of the program, please copy the entire folder of the project to avoid unexpected problems.
*   During the program run, if you need to terminate the program or `ffmpeg`, please press `Ctrl + C` to terminate the run, do not directly click the close button of the terminal window.

## Building Executables

### Build Executables Guide

This guide will guide you through forking this repository and executing GitHub Actions to automatically build and package programs based on the latest source code!

---

### Steps

#### 1. Fork this repository

1.  Click the **Fork** button in the upper right corner of the project repository to fork this repository to your personal GitHub account
2.  Your Fork repository address will be similar to: `https://github.com/your-username/this-repo`

---

#### 2. Enable GitHub Actions

1.  Go to your Fork repository page
2.  Click the **Settings** tab at the top
3.  Click the **Actions** tab on the right
4.  Click the **General** option
5.  Under **Actions permissions**, select the **Allow all actions and reusable workflows** option, and click the **Save** button

---

#### 3. Manually trigger the packaging process

1.  In your Fork repository, click the **Actions** tab at the top
2.  Find the workflow named **Build Executable**
3.  Click the **Run workflow** button on the right:
    *   Select the **master** or **develop** branch
    *   Click **Run workflow**

---

#### 4. View the packaging progress

1.  On the **Actions** page, you can see the trigger workflow run record
2.  Click the run record to view the detailed logs to understand the packaging progress and status

---

#### 5. Download the packaged result

1.  After the packaging is complete, go to the corresponding run record page
2.  In the **Artifacts** section at the bottom of the page, you will see the packaged result file
3.  Click to download and save locally, then you can get the packaged program

---

### Matters needing attention

1.  **Resource usage**:
    *   The Actions running environment is provided free of charge by GitHub. Ordinary users have a certain amount of free usage quota (2000 minutes) per month

2.  **Code modification**:
    *   You can freely modify the code in the Fork repository to customize the program packaging process
    *   After modifying, re-trigger the packaging process, and you will get a custom build version

3.  **Keep in sync with the main repository**:
    *   If the main repository updates the code or workflow, it is recommended that you synchronize the Fork repository regularly to get the latest features and fixes

---

### Actions Common Problems

#### Q1: Why can't I trigger the workflow?

A: Please confirm that you have followed the steps **Enable Actions**, otherwise GitHub will prohibit running the workflow

#### Q2: What should I do if the packaging process fails?

A:

*   Check the running logs to understand the reason for the failure
*   Make sure the code has no syntax errors or dependency issues
*   If the problem is still not resolved, you can raise the issue on the [Issues page](https://github.com/JoeanAmier/TikTokDownloader/issues) of this repository

#### Q3: Can I directly use the Actions of the main repository?

A: Due to permission restrictions, you cannot directly trigger the Actions of the main repository. Please execute the packaging process by forking the repository

## Program Update

<p>
    <b>Solution 1:</b> Download and unzip the file, and copy the old version of the `_internal\Volume` folder to the new version of the `_internal` folder.
</p>
<p>
    <b>Solution 2:</b> Download and unzip the file (do not run the program), copy all the files, and directly overwrite the old version files.
</p>

<hr>

## ⚠️ Disclaimer

<ol>
<li>The use of this project by the user is decided by the user himself and bears the risk by himself. The author is not responsible for any losses, responsibilities, or risks incurred by the user using this project.</li>
<li>The code and functions provided by the author of this project are based on the development results of existing knowledge and technology. The author strives to ensure the correctness and security of the code according to the current technical level, but does not guarantee that the code is completely free of errors or defects.</li>
<li>All third-party libraries, plug-ins, or services on which this project depends each follow their original open source or commercial licenses. The user needs to consult and abide by the corresponding agreements. The author does not assume any responsibility for the stability, security, and compliance of the third-party components.</li>
<li>When the user uses this project, he must strictly abide by the requirements of the <a href="https://github.com/JoeanAmier/TikTokDownloader/blob/master/LICENSE">GNU General Public License v3.0</a>, and indicate in the appropriate place that the code of the <a href="https://github.com/JoeanAmier/TikTokDownloader/blob/master/LICENSE">GNU General Public License v3.0</a> is used.</li>
<li>When the user uses the code and functions of this project, he must independently study the relevant laws and regulations, and ensure that his use behavior is legal and compliant. Any legal liability and risks arising from violations of laws and regulations shall be borne by the user himself.</li>
<li>The user may not use this tool to engage in any behavior that infringes intellectual property rights, including but not limited to unauthorized downloading and dissemination of copyrighted content. The developer does not participate in, support, or recognize the acquisition or distribution of any illegal content.</li>
<li>This project is not responsible for the compliance of the user's data collection, storage, transmission, and other processing activities. The user should abide by the relevant laws and regulations to ensure that the processing behavior is legal and proper; legal liabilities arising from violations of regulations shall be borne by the user himself.</li>
<li>Under no circumstances shall the user associate the author, contributors, or other relevant parties of this project with the user's use behavior, or require them to be responsible for any losses or damages incurred by the user's use of this project.</li>
<li>The author of this project will not provide paid versions of the DouK-Downloader project, nor will it provide any commercial services related to the DouK-Downloader project.</li>
<li>Any secondary development, modification, or compilation of the program based on this project is not related to the original author. The original author does not assume any responsibility related to the secondary development behavior or its results. The user should bear full responsibility for various situations that may be brought about by the secondary development.</li>
<li>This project does not grant any patent license to the user; if the use of this project leads to patent disputes or infringement, the user shall bear all risks and responsibilities. Without the written authorization of the author or the right holder, it is not allowed to use this project for any commercial promotion, promotion, or re-authorization.</li>
<li>The author reserves the right to terminate providing services to any user who violates this statement at any time, and may require them to destroy the code and derivative works that have been obtained.</li>
<li>The author reserves the right to update this statement without prior notice. The user's continued use is deemed to accept the revised terms.</li>
</ol>

<b>Before using the code and functions of this project, please carefully consider and accept the above disclaimer. If you have any questions or disagree with the above statement, please do not use the code and functions of this project. If you have used the code and functions of this project, it is deemed that you have fully understood and accepted the above disclaimer, and voluntarily bear all risks and consequences of using this project.</b>

<hr>
<h1>🌟 Contribution Guide</h1>

<p><b>Welcome to contribute to this project! In order to keep the code base clean, efficient and easy to maintain, please read the following guidelines carefully to ensure that your contributions can be accepted and integrated smoothly.</b></p>
<ul>
<li>Before starting development, please pull the latest code from the `develop` branch as the basis for modification; this helps to avoid merge conflicts and ensure that your changes are based on the latest project status.</li>
<li>If your changes involve multiple unrelated features or issues, please separate them into multiple independent commits or pull requests.</li>
<li>Each pull request should focus on a single function or fix as much as possible, so as to facilitate code review and testing.</li>
<li>Follow the existing code style; please make sure that your code is consistent with the existing code style in the project; it is recommended to use the Ruff tool to keep the code format specification.</li>
<li>Write readable code; add appropriate comments to help others understand your intentions.</li>
<li>Each commit should contain a clear and concise commit message to describe the changes made. Commit messages should follow the following format: `<Type>: <Short description>`</li>
<li>When you are ready to submit a pull request, please submit them to the `develop` branch first; this is to give the maintainers a buffer to perform additional testing and review before finally merging into the `master` branch.</li>
<li>It is recommended to communicate with the author before development or when you encounter doubts, to ensure that the development direction is consistent and avoid repeated labor or invalid submissions.</li>
</ul>
<p><b>Reference materials:</b></p>
<ul>
<li><a href="https://www.contributor-covenant.org/zh-cn/version/2/1/code_of_conduct/">Contributor Covenant</a></li>
<li><a href="https://opensource.guide/zh-hans/how-to-contribute/">How to contribute to open source</a></li>
</ul>

<hr>

# ♥️ Support the Project

<p>If <b>DouK-Downloader</b> is helpful to you, please consider giving it a <b>Star</b> ⭐, thank you for your support!</p>
<table>
<thead>
<tr>
<th align="center">WeChat</th>
<th align="center">Alipay</th>
</tr>
</thead>
<tbody><tr>
<td align="center"><img src="./docs/微信赞助二维码.png" alt="WeChat Sponsorship QR Code" height="200" width="200"></td>
<td align="center"><img src="./docs/支付宝赞助二维码.png" alt="Alipay Sponsorship QR Code" height="200" width="200"></td>
</tr>
</tbody>
</table>
<p>If you are willing, you can consider providing financial support for <b>DouK-Downloader</b>!</p>

<hr>

# 💰 Project Sponsorship

## DartNode

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")

***

## ZMTO

<a href="https://www.zmto.com/"><img src="https://console.zmto.com/templates/2019/dist/images/logo_dark.svg" alt="ZMTO"></a>
<p><a href="https://www.zmto.com/">ZMTO</a>: A professional cloud infrastructure provider, offering efficient solutions with reliable cutting-edge technology and professional support, and providing enterprise-level VPS infrastructure for eligible open-source projects, supporting the sustainable development and innovation of the open-source ecosystem.</p>

***

## TikHub

<p><a href="https://tikhub.io/">TikHub</a>: A third-party API service provider, focusing on providing APIs for various platforms.</p>
<p>By signing in daily, users can get a small amount of free usage; you can use my <b>Referral link</b>: <a href="https://user.tikhub.io/users/signup?referral_code=ZrdH8McC">https://user.tikhub.io/users/signup?referral_code=ZrdH8McC</a> or <b>Referral Code</b>: <code>ZrdH8McC</code>, register and recharge to get a `$2` credit!</p>

<hr>

# ✉️ Contact the Author

<ul>
<li>Author's Email: yonglelolu@foxmail.com</li>
<li>Author's WeChat: Downloader_Tools</li>
<li>WeChat Public Account: Downloader Tools</li>
<li><b>Discord Community</b>: <a href="https://discord.com/invite/ZYtmgKud9Y">Click to join the community</a></li>
<li>QQ Group Chat (project exchange): <a href="https://github.com/JoeanAmier/TikTokDownloader/blob/master/docs/QQ%E7%BE%A4%E8%81%8A%E4%BA%8C%E7%EF%BC%8F%81%E7%A0%81.png">Scan the QR code to join the group chat</a></li>
</ul>
<p>✨ <b>The author's other open source projects:</b></p>
<ul>
<li><b>XHS-Downloader (Xiaohongshu, RedNote)</b>: <a href="https://github.com/JoeanAmier/XHS-Downloader">https://github.com/JoeanAmier/XHS-Downloader</a></li>
<li><b>KS-Downloader (Kuaishou)</b>: <a href="https://github.com/JoeanAmier/KS-Downloader">https://github.com/JoeanAmier/KS-Downloader</a></li>
</ul>

<hr>

<h1>⭐ Star Trend</h1>
<p>
<img alt="Star History Chart" src="https://api.star-history.com/svg?repos=JoeanAmier/TikTokDownloader&amp;type=Timeline"/>
</p>

<hr>

# 💡 Project References

*   https://github.com/Johnserf-Seed/f2
*   https://github.com/Johnserf-Seed/TikTokDownload
*   https://github.com/Evil0ctal/Douyin_TikTok_Download_API
*   https://github.com/NearHuiwen/TiktokDouyinCrawler
*   https://github.com/ihmily/DouyinLiveRecorder
*   https://github.com/encode/httpx/
*   https://github.com/Textualize/rich
*   https://github.com/omnilib/aiosqlite
*   https://github.com/Tinche/aiofiles
*   https://github.com/thewh1teagle/rookie
*   https://github.com/pyinstaller/pyinstaller
*   https://foss.heptapod.net/openpyxl/openpyxl
*   https://github.com/carpedm20/emoji/
*   https://github.com/lxml/lxml
*   https://ffmpeg.org/ffmpeg-all.html
*   https://www.tikwm.com/