<div align="center">
<img src="./static/images/DouK-Downloader.png" alt="DouK-Downloader" height="256" width="256"><br>
<h1>DouK-Downloader</h1>
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

## Download TikTok and Douyin Videos Easily with DouK-Downloader

DouK-Downloader is a powerful and versatile open-source tool, based on the HTTPX module, for downloading videos, images, and data from TikTok and Douyin.  Access the original repository [here](https://github.com/JoeanAmier/TikTokDownloader).

**Key Features:**

*   ✅ **Multiple Platform Support:** Downloads from TikTok and Douyin.
*   ✅ **Various Content Types:** Supports videos, images, live streams, and collections.
*   ✅ **Account Data Download:**  Download videos from accounts by ID.
*   ✅ **High-Quality Downloads:**  Get videos in the highest available quality.
*   ✅ **Batch Downloading:** Download multiple videos at once.
*   ✅ **Data Saving:** Saves data in CSV/XLSX/SQLite formats.
*   ✅ **Proxy Support:** Supports using proxies for data collection.
*   ✅ **Web UI and API (Experimental):**  Web UI and API interfaces (currently under development).

<details>
<summary>Expanded Feature List</summary>

*   ✅ Download Douyin watermarked/unwatermarked videos/images
*   ✅ Download Douyin live streams/dynamic images
*   ✅ Download videos in the highest available quality
*   ✅ Download TikTok videos in original quality
*   ✅ Download TikTok unwatermarked videos/images
*   ✅ Download videos from Douyin accounts (posts/likes/favorites/collections)
*   ✅ Download videos from TikTok accounts (posts/likes)
*   ✅ Collect detailed Douyin/TikTok data
*   ✅ Batch download from links
*   ✅ Batch download from multiple accounts
*   ✅ Skip already downloaded files automatically
*   ✅ Save collected data permanently
*   ✅ Support CSV/XLSX/SQLite data formats
*   ✅ Download dynamic/static cover images
*   ✅ Get Douyin live stream addresses
*   ✅ Get TikTok live stream addresses
*   ✅ Use ffmpeg to download live streams
*   ✅ Web UI interactive interface
*   ✅ Collect Douyin comment data
*   ✅ Download Douyin collection works
*   ✅ Download TikTok collection works
*   ✅ Record statistics such as likes and favorites
*   ✅ Filter works by release time
*   ✅ Support incremental download of account works
*   ✅ Support using proxies for data collection
*   ✅ Support LAN remote access
*   ✅ Collect detailed Douyin account data
*   ✅ Update work statistics data
*   ✅ Support custom account/collection identifiers
*   ✅ Automatically update account nicknames/identifiers
*   ✅ Deploy to private servers
*   ✅ Deploy to public servers
*   ✅ Collect Douyin search data
*   ✅ Collect Douyin trending data
*   ✅ Record downloaded work IDs
*   ☑️ <del>Scan code login to get Cookie</del>
*   ✅ Read Cookie from the browser
*   ✅ Support Web API calls
*   ✅ Support multi-threaded download of works
*   ✅ File integrity processing mechanism
*   ✅ Custom rules for filtering works
*   ✅ Archive and save work files by folder
*   ✅ Customize file size upper limit
*   ✅ Support file breakpoint resume download
*   ✅ Monitor clipboard link download work

</details>

## Screenshots

<p><a href="https://www.bilibili.com/video/BV1d7eAzTEFs/">Watch the demo on bilibili</a>；<a href="https://youtu.be/yMU-RWl55hg">Watch the demo on YouTube</a></p>

### Terminal Interaction Mode

<p>It is recommended to manage accounts through configuration files. For more information, please refer to the <a href="https://github.com/JoeanAmier/TikTokDownloader/wiki/Documentation">documentation</a></p>

![Terminal Mode Screenshot](docs/screenshot/终端交互模式截图CN1.png)
*****
![Terminal Mode Screenshot](docs/screenshot/终端交互模式截图CN2.png)
*****
![Terminal Mode Screenshot](docs/screenshot/终端交互模式截图CN3.png)

### Web UI Interaction Mode

> **The project code has been refactored, and the code for this mode has not been updated. It will be re-opened when the development is completed in the future!**

### Web API Interface Mode

![WebAPI Mode Screenshot](docs/screenshot/WebAPI模式截图CN1.png)
*****
![WebAPI Mode Screenshot](docs/screenshot/WebAPI模式截图CN2.png)

> **After starting this mode, you can visit `http://127.0.0.1:5555/docs` or `http://127.0.0.1:5555/redoc` to view the automatically generated documentation!**

#### API Calling Example Code

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

## Getting Started

### Quick Start

<p>⭐ For Mac OS, Windows 10 and above users, go to <a href="https://github.com/JoeanAmier/TikTokDownloader/releases/latest">Releases</a> or <a href="https://github.com/JoeanAmier/TikTokDownloader/actions">Actions</a> to download the compiled program, and use it out of the box!</p>
<p>⭐ This project contains GitHub Actions that automatically build executable files. Users can use GitHub Actions to build the latest source code into executable files at any time!</p>
<p>⭐ Please refer to the <code>Building Executable File Guide</code> section of this document for the tutorial on automatically building executable files; if you need a more detailed graphic tutorial, please <a href="https://mp.weixin.qq.com/s/TorfoZKkf4-x8IBNLImNuw">refer to the article</a>!</p>
<p><strong>Note: The executable file <code>main</code> of the Mac OS platform may need to be started from the terminal command line; due to device limitations, the executable file of the Mac OS platform has not been tested, and its availability cannot be guaranteed!</strong></p>
<hr>
<ol>
<li><b>Run the executable file</b> or <b>Configure the environment to run</b>
<ol><b>Run the executable file</b>
<li>Download the executable file compressed package from <a href="https://github.com/JoeanAmier/TikTokDownloader/releases/latest">Releases</a> or Actions</li>
<li>After decompressing, open the program folder and double-click to run <code>main</code></li>
</ol>
<ol><b>Configure the environment to run</b>

[//]: # (<li>Install the <a href="https://www.python.org/">Python</a> interpreter with a version no less than <code>3.12</code></li>)
<li>Install the <a href="https://www.python.org/">Python</a> interpreter with version <code>3.12</code></li>
<li>Download the latest source code or the source code released by <a href="https://github.com/JoeanAmier/TikTokDownloader/releases/latest">Releases</a> to the local</li>
<li>Run the command <code>python -m venv venv</code> to create a virtual environment (optional)</li>
<li>Run the command <code>.\venv\Scripts\activate.ps1</code> or <code>venv\Scripts\activate</code> to activate the virtual environment (optional)</li>
<li>Run the command <code>pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt</code> to install the required modules for the program</li>
<li>Run the command <code>python .\main.py</code> or <code>python main.py</code> to start DouK-Downloader</li>
</ol>
</li>
<li>Read the disclaimer of DouK-Downloader and enter the content according to the prompt</li>
<li>Write the Cookie information to the configuration file
<ol><b>Read Cookie from the clipboard</b>
<li>Refer to the <a href="https://github.com/JoeanAmier/TikTokDownloader/blob/master/docs/Cookie%E8%8E%B7%E5%8F%96%E6%95%99%E7%A8%8B.md">Cookie extraction tutorial</a>, copy the required Cookie to the clipboard</li>
<li>Select the <code>Read Cookie from clipboard</code> option, and the program will automatically read the Cookie from the clipboard and write it to the configuration file</li>
</ol>
<ol><b>Read Cookie from the browser</b>
<li>Select the <code>Read Cookie from the browser</code> option and enter the browser type or number according to the prompt</li>
</ol>
<ol><b><del>Scan code login to get Cookie</del>（失效）</b>
<li><del>Select the <code>Scan code login to get Cookie</code> option, and the program will display the login QR code image, and open the image with the default application</del></li>
<li><del>Use the Douyin APP to scan the QR code and log in to the account</del></li>
<li><del>Follow the prompts to operate, and the program will automatically write the Cookie to the configuration file</del></li>
</ol>
</li>
<li>Return to the program interface, select <code>Terminal Interaction Mode</code> -> <code>Batch Download Link Works (General)</code> -> <code>Manually enter the works link to be collected</code> in sequence</li>
<li>Enter the Douyin work link to download the work file (the TikTok platform requires more initial settings, see the documentation for details)</li>
<li>For more detailed instructions, please refer to <b><a href="https://github.com/JoeanAmier/TikTokDownloader/wiki/Documentation">Project Documentation</a></b></li>
</ol>
<p>⭐ Recommended to use <a href="https://learn.microsoft.com/zh-cn/windows/terminal/install">Windows Terminal</a> (Windows 11 comes with the default terminal)</p>

### Docker Container

<ol>
<li>Get the image</li>
<ul>
<li>Method 1: Use the <code>Dockerfile</code> file to build the image</li>
<li>Method 2: Use the command <code>docker pull joeanamier/tiktok-downloader</code> to pull the image</li>
<li>Method 3: Use the command <code>docker pull ghcr.io/joeanamier/tiktok-downloader</code> to pull the image</li>
</ul>
<li>Create container: <code>docker run --name container name(optional) -p host port number:5555 -v tiktok_downloader_volume:/app/Volume -it &lt;image name&gt;</code>
</li>
<br><b>Note:</b> The <code>&lt;image name&gt;</code> here must be consistent with the image name you used in the first step (e.g. <code>joeanamier/tiktok-downloader</code> or <code>ghcr.io/joeanamier/tiktok-downloader</code>)
<li>Run the container
<ul>
<li>Start container: <code>docker start -i container name/container ID</code></li>
<li>Restart container: <code>docker restart -i container name/container ID</code></li>
</ul>
</li>
</ol>
<p>Docker containers cannot directly access the file system of the host machine, and some functions are not available, for example: <code>Read Cookie from the browser</code>; if there are exceptions in other functions, please feedback!</p>
<hr>

## About Cookie

[Click to view Cookie acquisition tutorial](https://github.com/JoeanAmier/TikTokDownloader/blob/master/docs/Cookie%E8%8E%B7%E5%8F%96%E6%95%99%E7%A8%8B.md)

> *   The Cookie only needs to be rewritten to the configuration file after it expires, not every time the program is run!
>
> *   The Cookie will affect the resolution of the downloaded video files. If you cannot download the highest resolution video files, please try to update the Cookie!
>
> *   If the program fails to get data, you can try to update the Cookie or use the logged-in Cookie!

<hr>

## Other Notes

<ul>
<li>When the program prompts the user to enter, pressing Enter directly means returning to the upper-level menu, and entering <code>Q</code> or <code>q</code> means ending the operation</li>
<li>Because getting account like works and favorite works data only returns the release date of the like/favorite works, and does not return the operation date, the program needs to get all the like/favorite works data before filtering by date; if there are many works, it may take a long time; can be controlled through the <code>max_pages</code> parameter number of requests</li>
<li>Getting the published works data of a private account requires the Cookie after login, and the logged-in account needs to follow the private account</li>
<li>When batch downloading account works or collection works, if the corresponding nickname or identifier changes, the program will automatically update the nickname and identifier in the downloaded work file name</li>
<li>When the program downloads files, it will first download the files to a temporary folder, and after the download is complete, it will move them to the storage folder; the temporary folder will be emptied when the program ends</li>
<li><code>Batch Download Favorite Works Mode</code> currently only supports downloading favorite works of the account corresponding to the logged-in Cookie, and does not support multiple accounts</li>
<li>If you want the program to use a proxy to request data, you must set the <code>proxy</code> parameter in <code>settings.json</code>, otherwise the program will not use the proxy</li>
<li>If your computer does not have a suitable program to edit the JSON file, it is recommended to use the <a href="https://try8.cn/tool/format/json">JSON online tool</a> to edit the configuration file content</li>
<li>When the program requests the user to enter content or links, please avoid the content or links entered contain line breaks, which may cause unexpected problems</li>
<li>This project will not support the download of paid works, please do not feedback any questions about the download of paid works</li>
<li>Windows systems need to run the program as an administrator to read the Cookie of Chromium, Chrome, Edge browsers</li>
<li>This project has not been optimized for multiple program instances. If you need multiple program instances, please copy the entire project folder to avoid unexpected problems</li>
<li>During the operation of the program, if you need to terminate the program or <code>ffmpeg</code>, please press <code>Ctrl + C</code> to terminate the operation, and do not directly click the close button of the terminal window</li>
</ul>
<h2>Building Executable File Guide</h2>
<details>
<summary><b>Building Executable File Guide（Click to expand）</b></summary>

This guide will guide you through forking this repository and automatically completing program building and packaging based on the latest source code by executing GitHub Actions!

---

### Steps to use

#### 1. Fork this repository

1.  Click the **Fork** button in the upper right corner of the project repository to fork this repository to your personal GitHub account
2.  Your Fork repository address will be similar to: `https://github.com/your-username/this-repo`

---

#### 2. Enable GitHub Actions

1.  Go to the page of your Fork repository
2.  Click the **Settings** tab at the top
3.  Click the **Actions** tab on the right
4.  Click the **General** option
5.  In the **Actions permissions**, select the **Allow all actions and reusable workflows** option, and click the **Save** button

---

#### 3. Manually trigger the packaging process

1.  In the repository you Fork, click the **Actions** tab at the top
2.  Find the workflow named **Build Executable File**
3.  Click the **Run workflow** button on the right:
    -   Select **master** or **develop** branch
    -   Click **Run workflow**

---

#### 4. Check the packaging progress

1.  On the **Actions** page, you can see the run records of the triggered workflow
2.  Click the run record to view the detailed logs to understand the packaging progress and status

---

#### 5. Download the packaging result

1.  After the packaging is complete, go to the corresponding run record page
2.  In the **Artifacts** section at the bottom of the page, you will see the packaged result file
3.  Click to download and save locally, you can get the packaged program

---

### Matters needing attention

1.  **Resource usage**:
    -   The running environment of Actions is provided free of charge by GitHub. Ordinary users have a certain free usage quota per month (2000 minutes)

2.  **Code modification**:
    -   You can freely modify the code in the Fork repository to customize the program packaging process
    -   After modification, trigger the packaging process again, and you will get the customized build version

3.  **Keep synchronized with the main repository**:
    -   If the main repository updates the code or workflow, it is recommended that you regularly synchronize the Fork repository to get the latest features and fixes

---

### Actions FAQs

#### Q1: Why can't I trigger the workflow?

A: Please make sure you have followed the steps to **enable Actions**, otherwise GitHub will prohibit running the workflow

#### Q2: What should I do if the packaging process fails?

A:

-   Check the run log to understand the cause of the failure
-   Make sure the code has no syntax errors or dependency problems
-   If the problem is still unresolved, you can raise the question on the [Issues page](https://github.com/JoeanAmier/TikTokDownloader/issues) of this repository

#### Q3: Can I directly use the Actions of the main repository?

A: Due to permission restrictions, you cannot directly trigger the Actions of the main repository. Please execute the packaging process through the Fork repository

</details>

## Program Updates

<p><strong>Method 1:</strong> Download and unzip the file, and copy the old version of the <code>_internal\Volume</code> folder to the new version of the <code>_internal</code> folder.</p>
<p><strong>Method 2:</strong> Download and unzip the file (do not run the program), copy all the files, and directly overwrite the old version files.</p>

# ⚠️ Disclaimer

<ol>
<li>The user's use of this project is decided by the user, and the user shall bear the risk. The author is not responsible for any loss, liability, or risk caused by the user's use of this project.</li>
<li>The code and functions provided by the author of this project are based on the development results of existing knowledge and technology. The author strives to ensure the correctness and security of the code according to the existing technical level, but does not guarantee that the code is completely free of errors or defects.</li>
<li>All third-party libraries, plug-ins or services that this project relies on follow their original open source or commercial licenses. Users need to check and abide by the corresponding agreements, and the author does not assume any responsibility for the stability, security, and compliance of third-party components.</li>
<li>When users use this project, they must strictly abide by the requirements of <a href="https://github.com/JoeanAmier/TikTokDownloader/blob/master/LICENSE">GNU General Public License v3.0</a>, and indicate the use of <a href="https://github.com/JoeanAmier/TikTokDownloader/blob/master/LICENSE">GNU General Public License v3.0</a> code in appropriate places.</li>
<li>When users use the code and functions of this project, they must study the relevant laws and regulations by themselves and ensure that their use behavior is legal and compliant. Any legal responsibility and risk caused by violation of laws and regulations shall be borne by the user.</li>
<li>Users shall not use this tool to engage in any activities that infringe intellectual property rights, including but not limited to unauthorized downloading and dissemination of content protected by copyright. Developers do not participate in, support, or recognize the acquisition or distribution of any illegal content.</li>
<li>This project does not assume responsibility for the compliance of the user's data collection, storage, transmission, and other processing activities. Users should abide by relevant laws and regulations to ensure that the processing behavior is legal and correct; the legal responsibility caused by illegal operations shall be borne by the user.</li>
<li>Users shall not, under any circumstances, associate the author, contributors, or other relevant parties of this project with the user's use behavior, or require them to be responsible for any loss or damage caused by the user's use of this project.</li>
<li>The author of this project will not provide the paid version of the DouK-Downloader project, nor will it provide any commercial services related to the DouK-Downloader project.</li>
<li>Any secondary development, modification, or compilation of the program based on this project has nothing to do with the original author. The original author does not assume any responsibility related to the secondary development behavior or its results. Users should bear all responsibilities for various situations that may be caused by secondary development.</li>
<li>This project does not grant users any patent licenses; if the use of this project leads to patent disputes or infringement, the user shall bear all risks and responsibilities by themselves. Without the written authorization of the author or the right holder, the project shall not be used for any commercial promotion, promotion, or reauthorization.</li>
<li>The author reserves the right to terminate the service to any user who violates this statement at any time, and may require them to destroy the code and derivative works that have been obtained.</li>
<li>The author reserves the right to update this statement without prior notice, and the user's continued use is deemed to accept the revised terms.</li>
</ol>
<b>Before using the code and functions of this project, please carefully consider and accept the above disclaimer. If you have any questions or disagree with the above statement, please do not use the code and functions of this project. If you use the code and functions of this project, it is deemed that you have fully understood and accepted the above disclaimer, and voluntarily assume all risks and consequences of using this project.</b>
<h1>🌟 Contribution Guide</h1>
<p><strong>Welcome to contribute to this project! In order to keep the code base clean, efficient, and easy to maintain, please read the following guidelines carefully to ensure that your contributions can be accepted and integrated smoothly.</strong></p>
<ul>
<li>Before starting development, please pull the latest code from the <code>develop</code> branch as the basis for modification; this helps to avoid merge conflicts and ensure that your changes are based on the latest project status.</li>
<li>If your changes involve multiple unrelated functions or issues, please divide them into multiple independent submissions or pull requests.</li>
<li>Each pull request should focus on a single function or fix as much as possible to facilitate code review and testing.</li>
<li>Follow the existing code style; please make sure your code is consistent with the existing code style in the project; it is recommended to use the Ruff tool to maintain the code format specifications.</li>
<li>Write readable code; add appropriate comments to help others understand your intentions.</li>
<li>Each submission should contain a clear and concise submission message to describe the changes made. The submission message should follow the following format: <code>&lt;type&gt;: &lt;short description&gt;</code></li>
<li>When you are ready to submit a pull request, please give priority to submitting them to the <code>develop</code> branch; this is to give the maintainer a buffer zone for additional testing and review before finally merging into the <code>master</code> branch.</li>
<li>It is recommended to communicate with the author before development or when encountering questions to ensure consistency of development direction and avoid duplicate work or invalid submissions.</li>
</ul>
<p><strong>Reference materials:</strong></p>
<ul>
<li><a href="https://www.contributor-covenant.org/zh-cn/version/2/1/code_of_conduct/">Contributor Covenant</a></li>
<li><a href="https://opensource.guide/zh-hans/how-to-contribute/">How to contribute to open source</a></li>
</ul>

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
<p>If you are willing, you can consider providing funding for <b>DouK-Downloader</b> to provide additional support!</p>

# 💰 Project Sponsorship

## DartNode

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")

***

## ZMTO

<a href="https://www.zmto.com/"><img src="https://console.zmto.com/templates/2019/dist/images/logo_dark.svg" alt="ZMTO"></a>
<p><a href="https://www.zmto.com/">ZMTO</a>: A professional cloud infrastructure provider, providing efficient solutions with reliable cutting-edge technology and professional support, and providing enterprise-level VPS infrastructure for eligible open source projects, supporting the sustainable development and innovation of the open source ecosystem.</p>

***

## TikHub

<p><a href="https://tikhub.io/">TikHub</a>: A third-party API service provider, focusing on providing API for various platforms.</p>
<p>By signing in daily, users can get a small amount of usage; you can use my <strong>recommendation link</strong>: <a href="https://user.tikhub.io/users/signup?referral_code=ZrdH8McC">https://user.tikhub.io/users/signup?referral_code=ZrdH8McC</a> or <strong>recommendation code</strong>: <code>ZrdH8McC</code>, register and recharge to get <code>$2</code> credits!</p>

# ✉️ Contact the Author

<ul>
<li>Author's email: yonglelolu@foxmail.com</li>
<li>Author's WeChat: Downloader_Tools</li>
<li>WeChat official account: Downloader Tools</li>
<li><b>Discord Community</b>: <a href="https://discord.com/invite/ZYtmgKud9Y">Click to join the community</a></li>
<li>QQ Group Chat (project communication): <a href="https://github.com/JoeanAmier/TikTokDownloader/blob/master/docs/QQ%E7%BE%A4%E8%81%8A%E4%BA%8C%E7%BB%B4%E7%A0%81.png">Scan the QR code to join the group chat</a></li>
</ul>
<p>✨ <b>Other open source projects of the author:</b></p>
<ul>
<li><b>XHS-Downloader (XiaoHongShu, RedNote)</b>: <a href="https://github.com/JoeanAmier/XHS-Downloader">https://github.com/JoeanAmier/XHS-Downloader</a></li>
<li><b>KS-Downloader (KuaiShou)</b>: <a href="https://github.com/JoeanAmier/KS-Downloader">https://github.com/JoeanAmier/KS-Downloader</a></li>
</ul>
<h1>⭐ Star Trend</h1>
<p>
<img alt="Star History Chart" src="https://api.star-history.com/svg?repos=JoeanAmier/TikTokDownloader&amp;type=Timeline"/>
</p>

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
```
Key improvements and SEO optimization:

*   **Headline:** Includes primary keywords "TikTok" and "Douyin" (which is the Chinese name for TikTok).
*   **Hook:** A concise and engaging opening sentence.
*   **Key Features:**  Uses bullet points for easy readability, and includes keywords for what users will search for.
*   **Clear Structure:** Uses headings, subheadings, and details elements to organize the information.
*   **Keyword Density:** The most important keywords are repeated naturally throughout the text.
*   **Call to Action:** Encourages the user to "star" the project.
*   **Stronger Emphasis:** Use of bolding and *italics*.
*   **Link Back:**  Includes a prominent link back to the original GitHub repository.
*   **Screenshots and Visuals:** Includes screenshots to enhance the user experience.
*   **Clear Instructions:**  Provides a quick start guide, detailed steps, and troubleshooting advice.
*   **Contribution and Disclaimer:** Emphasizes contribution guidelines and the disclaimer.
*   **Contact Information:** Includes multiple contact methods.
*   **Project References:** List relevant project references.