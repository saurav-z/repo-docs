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

## DouK-Downloader: Your All-in-One TikTok and Douyin Video Downloader

Tired of watermarks and limitations? **DouK-Downloader empowers you to download your favorite TikTok and Douyin content with ease and flexibility.**  Find the original repo [here](https://github.com/JoeanAmier/TikTokDownloader).

**Key Features:**

*   ✅ **Comprehensive Downloading:** Download videos, images, and more from TikTok and Douyin, including posts, likes, collections, and live streams.
*   ✅ **High-Quality Options:** Get videos in their original quality, without watermarks.
*   ✅ **Account & Collection Downloads:** Bulk download content from user accounts and collections.
*   ✅ **Flexible Data Handling:** Save data in CSV, XLSX, or SQLite formats.
*   ✅ **Proxy Support:** Use proxies for data collection.
*   ✅ **Web UI and API (Coming Soon):**  Explore interactive and programmatic access.
*   ✅ **Automated Features:**  Includes features like automatic file naming, skipping downloaded files, and clipboard monitoring.
*   ✅ **Cross-Platform Support**: Works with Windows, macOS and Docker

<details>
<summary>Function List</summary>
<ul>
<li>✅ Download Douyin/TikTok videos & images</li>
<li>✅ Download Douyin/TikTok Live Videos</li>
<li>✅ High Quality downloads</li>
<li>✅ Download Douyin/TikTok account content</li>
<li>✅ Gather detailed account data</li>
<li>✅ Batch Downloads</li>
<li>✅ Skip already downloaded files</li>
<li>✅ Save data to CSV/XLSX/SQLite</li>
<li>✅ Dynamic cover image downloads</li>
<li>✅ Get Douyin/TikTok live stream addresses</li>
<li>✅ Docker support</li>
</ul>
</details>

## 💻 Program Screenshots

<p><a href="https://www.bilibili.com/video/BV1d7eAzTEFs/">Watch a demonstration on Bilibili</a>; <a href="https://youtu.be/yMU-RWl55hg">Watch a demonstration on YouTube</a></p>

### Terminal Mode

<p>It is recommended to manage accounts via configuration files. More information can be found in the <a href="https://github.com/JoeanAmier/TikTokDownloader/wiki/Documentation">documentation</a></p>

![Terminal Mode Screenshot 1](docs/screenshot/终端交互模式截图CN1.png)
*****
![Terminal Mode Screenshot 2](docs/screenshot/终端交互模式截图CN2.png)
*****
![Terminal Mode Screenshot 3](docs/screenshot/终端交互模式截图CN3.png)

### Web UI Mode (Under Development)

> **This mode is currently under development.  It will be re-opened once the code has been updated!**

### Web API Interface Mode

![WebAPI Mode Screenshot 1](docs/screenshot/WebAPI模式截图CN1.png)
*****
![WebAPI Mode Screenshot 2](docs/screenshot/WebAPI模式截图CN2.png)

> **Start this mode, then visit `http://127.0.0.1:5555/docs` or `http://127.0.0.1:5555/redoc` to view the automatically generated documentation!**

#### API Call Example Code

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

## 📋 Project Information

### Quick Start

<p>⭐ Mac OS, Windows 10 and above users can download the compiled program from <a href="https://github.com/JoeanAmier/TikTokDownloader/releases/latest">Releases</a> or <a href="https://github.com/JoeanAmier/TikTokDownloader/actions">Actions</a>, and use it out of the box!</p>
<p>⭐ This project includes GitHub Actions that automatically build executable files. Users can use GitHub Actions to build the latest source code into executable files at any time!</p>
<p>⭐ For a tutorial on automatically building executable files, please refer to the <code>Guide to Building Executable Files</code> section of this document. If you need a more detailed tutorial with images, please <a href="https://mp.weixin.qq.com/s/TorfoZKkf4-x8IBNLImNuw">refer to the article</a>!</p>
<p><strong>Note: The executable file <code>main</code> on the Mac OS platform may need to be started from the terminal command line; due to device limitations, the executable file on the Mac OS platform has not been tested and its availability cannot be guaranteed!</strong></p>
<hr>
<ol>
<li><b>Run the executable file</b> or <b>Configure the environment to run</b>
<ol><b>Run the executable file</b>
<li>Download the executable file compressed package from <a href="https://github.com/JoeanAmier/TikTokDownloader/releases/latest">Releases</a> or Actions</li>
<li>Unzip and open the program folder, double-click and run <code>main</code></li>
</ol>
<ol><b>Configure the environment to run</b>

[//]: # (<li>Install a <a href="https://www.python.org/">Python</a> interpreter not lower than <code>3.12</code></li>)
<li>Install a <code>3.12</code> version of the <a href="https://www.python.org/">Python</a> interpreter</li>
<li>Download the latest source code or the source code released by <a href="https://github.com/JoeanAmier/TikTokDownloader/releases/latest">Releases</a> to the local</li>
<li>Run the <code>python -m venv venv</code> command to create a virtual environment (optional)</li>
<li>Run the <code>.\venv\Scripts\activate.ps1</code> or <code>venv\Scripts\activate</code> command to activate the virtual environment (optional)</li>
<li>Run the <code>pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt</code> command to install the modules required by the program</li>
<li>Run the <code>python .\main.py</code> or <code>python main.py</code> command to start DouK-Downloader</li>
</ol>
</li>
<li>Read the DouK-Downloader disclaimer and enter the content as prompted</li>
<li>Write Cookie information to the configuration file
<ol><b>Read Cookie from clipboard</b>
<li>Refer to the <a href="https://github.com/JoeanAmier/TikTokDownloader/blob/master/docs/Cookie%E8%8E%B7%E5%8F%96%E6%95%99%E7%A8%8B.md">Cookie extraction tutorial</a>, copy the required Cookie to the clipboard</li>
<li>Select the <code>Read Cookie from clipboard</code> option, and the program will automatically read the Cookie from the clipboard and write it to the configuration file</li>
</ol>
<ol><b>Read Cookie from browser</b>
<li>Select the <code>Read Cookie from browser</code> option, and enter the browser type or number as prompted</li>
</ol>
<ol><b><del>Scan the code to log in to get Cookie</del> (invalid)</b>
<li><del>Select the <code>Scan the code to log in to get Cookie</code> option, the program will display a login QR code image, and use the default application to open the image</del></li>
<li><del>Use the Douyin APP to scan the QR code and log in to the account</del></li>
<li><del>Follow the prompts to operate, and the program will automatically write the Cookie to the configuration file</del></li>
</ol>
</li>
<li>Return to the program interface, select <code>Terminal interaction mode</code> -> <code>Batch download link works (general)</code> -> <code>Manually enter the works link to be collected</code> in sequence</li>
<li>Enter the Douyin work link to download the work file (TikTok platform needs more initial settings, see the documentation for details)</li>
<li>For more detailed instructions, please refer to the <b><a href="https://github.com/JoeanAmier/TikTokDownloader/wiki/Documentation">project documentation</a></b></li>
</ol>
<p>⭐ It is recommended to use <a href="https://learn.microsoft.com/zh-cn/windows/terminal/install">Windows Terminal</a> (Windows 11 comes with a default terminal)</p>

### Docker Container

<ol>
<li>Get image</li>
<ul>
<li>Method 1: Use the <code>Dockerfile</code> file to build the image</li>
<li>Method 2: Use the <code>docker pull joeanamier/tiktok-downloader</code> command to pull the image</li>
<li>Method 3: Use the <code>docker pull ghcr.io/joeanamier/tiktok-downloader</code> command to pull the image</li>
</ul>
<li>Create container: <code>docker run --name container name(optional) -p host port number:5555 -v tiktok_downloader_volume:/app/Volume -it &lt;image name&gt;</code>
</li>
<br><b>Note:</b> The <code>&lt;image name&gt;</code> here needs to be consistent with the image name you used in the first step (e.g. <code>joeanamier/tiktok-downloader</code> or <code>ghcr.io/joeanamier/tiktok-downloader</code>)
<li>Run the container
<ul>
<li>Start the container: <code>docker start -i container name/container ID</code></li>
<li>Restart the container: <code>docker restart -i container name/container ID</code></li>
</ul>
</li>
</ol>
<p>Docker containers cannot directly access the file system of the host machine, so some functions are not available, such as: <code>Read Cookie from browser</code>; if there are other function exceptions, please feedback!</p>
<hr>

## About Cookies

[Click to view the Cookie tutorial](https://github.com/JoeanAmier/TikTokDownloader/blob/master/docs/Cookie%E8%8E%B7%E5%8F%96%E6%95%99%E7%A8%8B.md)

> * You only need to rewrite the configuration file after the Cookie expires, not every time you run the program!
>
> * Cookies affect the resolution of the downloaded video files. If you cannot download video files with the highest resolution, please try to update the Cookie!
>
> * When the program fails to get the data, you can try to update the Cookie or use the logged-in Cookie!

<hr>

## Other Notes

<ul>
<li>When the program prompts the user to enter, pressing Enter directly means returning to the upper menu, and entering <code>Q</code> or <code>q</code> means ending the run</li>
<li>Since obtaining account favorite works and collection works data only returns the release date of the favorite/collection works, and does not return the operation date, the program needs to obtain all the favorite/collection works data before performing date filtering; if the number of works is large, it may take a long time; the <code>max_pages</code> parameter can be used to control the number of requests</li>
<li>Obtaining the release work data of private accounts requires the logged-in Cookie, and the logged-in account needs to follow the private account</li>
<li>When batch downloading account works or collection works, if the corresponding nickname or identification changes, the program will automatically update the nickname and identification in the downloaded work file name</li>
<li>When the program downloads files, it will first download the files to a temporary folder, and then move them to the storage folder after the download is complete; the program will clear the temporary folder at the end of the run</li>
<li><code>Batch download collection works mode</code> currently only supports downloading the collection works of the account corresponding to the currently logged-in Cookie, and does not support multiple accounts</li>
<li>If you want the program to use a proxy to request data, you must set the <code>proxy</code> parameter in <code>settings.json</code>, otherwise the program will not use the proxy</li>
<li>If your computer does not have a suitable program to edit JSON files, it is recommended to use <a href="https://try8.cn/tool/format/json">JSON online tool</a> to edit the configuration file content</li>
<li>When the program requests the user to enter content or a link, please avoid the content or link entered containing line breaks, which may cause unexpected problems</li>
<li>This project will not support the download of paid works, please do not feed back any questions about the download of paid works</li>
<li>The Windows system needs to run the program as an administrator to read the Chromium, Chrome, and Edge browser Cookies</li>
<li>This project has not been optimized for the case of multiple programs being opened. If you need to open multiple programs, please copy the entire project folder to avoid unexpected problems</li>
<li>During the program run, if you need to terminate the program or <code>ffmpeg</code>, please press <code>Ctrl + C</code> to terminate the run, do not directly click the close button of the terminal window</li>
</ul>

<h2>Guide to Building Executable Files</h2>
<details>
<summary><b>Guide to Building Executable Files (Click to Expand)</b></summary>

This guide will guide you to automatically complete the program building and packaging based on the latest source code by forking this repository and executing GitHub Actions!

---

### Steps of Use

#### 1. Fork this repository

1.  Click the **Fork** button in the upper right corner of the project repository to fork this repository to your personal GitHub account
2.  Your Fork repository address will be similar to: `https://github.com/your-username/this-repo`

---

#### 2. Enable GitHub Actions

1.  Go to the page of your Fork repository
2.  Click the **Settings** tab at the top
3.  Click the **Actions** tab on the right
4.  Click the **General** option
5.  Under **Actions permissions**, select the **Allow all actions and reusable workflows** option, and click the **Save** button

---

#### 3. Manually Trigger the Packaging Process

1.  In your Fork repository, click the **Actions** tab at the top
2.  Find the workflow named **构建可执行文件**
3.  Click the **Run workflow** button on the right:
    -   Select the **master** or **develop** branch
    -   Click **Run workflow**

---

#### 4. View the Packaging Progress

1.  On the **Actions** page, you can see the workflow running record triggered
2.  Click the running record to view the detailed logs to understand the packaging progress and status

---

#### 5. Download the Packaging Results

1.  After the packaging is complete, enter the corresponding running record page
2.  In the **Artifacts** section at the bottom of the page, you will see the packaged result files
3.  Click to download and save it locally, you can get the packaged program

---

### Notes

1.  **Resource Use**:
    -   The Actions running environment is provided by GitHub for free, and ordinary users have a certain amount of free use allowance per month (2000 minutes)

2.  **Code Modification**:
    -   You can freely modify the code in the Fork repository to customize the program packaging process
    -   After modification, re-trigger the packaging process, you will get the customized build version

3.  **Keep in Sync with the Main Repository**:
    -   If the main repository updates the code or workflow, it is recommended that you synchronize the Fork repository regularly to get the latest features and fixes

---

### Actions FAQs

#### Q1: Why can't I trigger the workflow?

A: Please confirm that you have followed the steps to **Enable Actions**, otherwise GitHub will prohibit running the workflow

#### Q2: What if the packaging process fails?

A:

-   Check the running logs to understand the cause of the failure
-   Make sure the code has no syntax errors or dependency issues
-   If the problem is still not solved, you can raise the issue on the [Issues page](https://github.com/JoeanAmier/TikTokDownloader/issues) of this repository

#### Q3: Can I directly use the Actions of the main repository?

A: Due to permission restrictions, you cannot directly trigger the Actions of the main repository. Please execute the packaging process by forking the repository

</details>

## Program Updates

<p><strong>Method 1:</strong> Download and unzip the file, and copy the old version <code>_internal\Volume</code> folder to the new version <code>_internal</code> folder.</p>
<p><strong>Method 2:</strong> Download and unzip the file (do not run the program), copy all the files, and directly overwrite the old version files.</p>

# ⚠️ Disclaimer

<ol>
<li>The use of this project by the user is decided by the user and the user assumes the risk. The author is not responsible for any losses, liabilities, or risks arising from the use of this project by the user.</li>
<li>The code and functions provided by the author of this project are based on the results of existing knowledge and technology development. The author strives to ensure the correctness and security of the code according to the existing technical level, but does not guarantee that the code is completely free of errors or defects.</li>
<li>All third-party libraries, plug-ins, or services on which this project depends each follow their original open source or commercial licenses. The user should check and abide by the corresponding agreement on their own. The author does not assume any responsibility for the stability, security, and compliance of third-party components.</li>
<li>When using this project, the user must strictly abide by the requirements of <a href="https://github.com/JoeanAmier/TikTokDownloader/blob/master/LICENSE">GNU General Public License v3.0</a> and indicate in the appropriate place that the code of <a href="https://github.com/JoeanAmier/TikTokDownloader/blob/master/LICENSE">GNU General Public License v3.0</a> is used.</li>
<li>When using the code and functions of this project, the user must study the relevant laws and regulations on their own and ensure that their use behavior is legal and compliant. Any legal liabilities and risks arising from the violation of laws and regulations shall be borne by the user.</li>
<li>The user shall not use this tool to engage in any behavior that infringes intellectual property rights, including but not limited to unauthorized downloading and dissemination of content protected by copyright. The developer does not participate in, does not support, and does not recognize the acquisition or distribution of any illegal content.</li>
<li>This project is not responsible for the compliance of data collection, storage, transmission, and other processing activities involved by the user. The user should abide by relevant laws and regulations on their own and ensure that the processing behavior is legal and proper; the legal liabilities arising from the illegal operation shall be borne by the user.</li>
<li>The user shall not, under any circumstances, associate the author, contributors, or other relevant parties of this project with the user's use behavior, or ask them to be responsible for any losses or damages arising from the user's use of this project.</li>
<li>The author of this project will not provide paid versions of the DouK-Downloader project, nor will it provide any commercial services related to the DouK-Downloader project.</li>
<li>Any secondary development, modification, or compilation of the program based on this project is not related to the original author. The original author does not assume any responsibility related to the secondary development behavior or its results. The user should bear all the responsibilities for various situations that may be brought about by secondary development.</li>
<li>This project does not grant any patent license to the user; if the use of this project leads to patent disputes or infringement, the user shall bear all risks and responsibilities on their own. Without the written authorization of the author or the right holder, the project shall not be used for any commercial promotion, promotion, or re-authorization.</li>
<li>The author reserves the right to terminate the service to any user who violates this statement at any time and may require the user to destroy the code and derivative works that have been obtained.</li>
<li>The author reserves the right to update this statement without prior notice, and the continued use of the user is deemed to accept the revised terms.</li>
</ol>
<b>Before using the code and functions of this project, please carefully consider and accept the above disclaimer. If you have any questions or disagree with the above statement, please do not use the code and functions of this project. If you use the code and functions of this project, it is deemed that you have fully understood and accepted the above disclaimer, and voluntarily assume all risks and consequences of using this project.</b>

<h1>🌟 Contribution Guide</h1>
<p><strong>Welcome to contribute to this project! To keep the code base clean, efficient, and easy to maintain, please read the following guidelines carefully to ensure that your contribution can be accepted and integrated smoothly.</strong></p>
<ul>
<li>Before starting development, please pull the latest code from the <code>develop</code> branch and use it as the basis for modification; this will help avoid merge conflicts and ensure that your changes are based on the latest project status.</li>
<li>If your changes involve multiple unrelated functions or issues, please divide them into multiple independent commits or pull requests.</li>
<li>Each pull request should focus on a single function or fix as much as possible for easy code review and testing.</li>
<li>Follow the existing code style; please make sure that your code is consistent with the existing code style in the project; it is recommended to use the Ruff tool to maintain the code format specifications.</li>
<li>Write readable code; add appropriate comments to help others understand your intent.</li>
<li>Each commit should contain a clear and concise commit message to describe the changes made. The commit message should follow the following format: <code>&lt;type&gt;: &lt;brief description&gt;</code></li>
<li>When you are ready to submit a pull request, please prioritize submitting them to the <code>develop</code> branch; this is to give the maintainers a buffer for additional testing and review before the final merge into the <code>master</code> branch.</li>
<li>It is recommended to communicate with the author before development or when encountering doubts to ensure that the development direction is consistent, and to avoid repetitive work or invalid submissions.</li>
</ul>
<p><strong>References:</strong></p>
<ul>
<li><a href="https://www.contributor-covenant.org/zh-cn/version/2/1/code_of_conduct/">Contributor Covenant</a></li>
<li><a href="https://opensource.guide/zh-hans/how-to-contribute/">How to Contribute to Open Source</a></li>
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
<p>If you are willing, you can consider providing funding to provide additional support for <b>DouK-Downloader</b>!</p>

# 💰 Project Sponsorship

## DartNode

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")

***

## ZMTO

<a href="https://www.zmto.com/"><img src="https://console.zmto.com/templates/2019/dist/images/logo_dark.svg" alt="ZMTO"></a>
<p><a href="https://www.zmto.com/">ZMTO</a>: A professional cloud infrastructure provider that provides efficient solutions with reliable cutting-edge technology and professional support, and provides enterprise-level VPS infrastructure for eligible open source projects to support the sustainable development and innovation of the open source ecosystem.</p>

***

## TikHub

<p><a href="https://tikhub.io/">TikHub</a>: A third-party API service provider, focusing on providing API for various platforms.</p>
<p>By signing in daily, users can get a small amount of usage credit for free; you can use my <strong>recommendation link</strong>: <a href="https://user.tikhub.io/users/signup?referral_code=ZrdH8McC">https://user.tikhub.io/users/signup?referral_code=ZrdH8McC</a> or <strong>recommendation code</strong>: <code>ZrdH8McC</code>, register and recharge to get <code>$2</code> credit!</p>

# ✉️ Contact the Author

<ul>
<li>Author's email: yonglelolu@foxmail.com</li>
<li>Author's WeChat: Downloader_Tools</li>
<li>WeChat official account: Downloader Tools</li>
<li><b>Discord community</b>: <a href="https://discord.com/invite/ZYtmgKud9Y">Click to join the community</a></li>
<li>QQ Group Chat (Project Exchange): <a href="https://github.com/JoeanAmier/TikTokDownloader/blob/master/docs/QQ%E7%BE%A4%E8%81%8A%E4%BA%8C%E7%BB%B4%E7%A0%81.png">Scan the QR code to join the group chat</a></li>
</ul>
<p>✨ <b>Other open source projects by the author:</b></p>
<ul>
<li><b>XHS-Downloader (Xiaohongshu, RedNote)</b>: <a href="https://github.com/JoeanAmier/XHS-Downloader">https://github.com/JoeanAmier/XHS-Downloader</a></li>
<li><b>KS-Downloader (Kuaishou)</b>: <a href="https://github.com/JoeanAmier/KS-Downloader">https://github.com/JoeanAmier/KS-Downloader</a></li>
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