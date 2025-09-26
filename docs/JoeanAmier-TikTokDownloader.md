<div align="center">
<img src="./static/images/DouK-Downloader.png" alt="DouK-Downloader" height="256" width="256"><br>
<h1>DouK-Downloader: Download TikTok and Douyin Videos Easily</h1>
<p>🔥 Unleash the power to download TikTok and Douyin videos, livestreams, and data with DouK-Downloader, an open-source, Python-based tool. <a href="https://github.com/JoeanAmier/TikTokDownloader">Explore the repository.</a></p>

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

*   ✅ Download Douyin & TikTok videos (no watermarks), images, and collections
*   ✅ Download high-quality video files
*   ✅ Download videos from liked, published, and favorite sections
*   ✅ Collect data from accounts
*   ✅ Download via links
*   ✅ Collect Douyin/TikTok account details
*   ✅ Get live streaming addresses
*   ✅ Support web API calls

# 💻 Program Screenshots

<p><a href="https://www.bilibili.com/video/BV1d7eAzTEFs/">Go to bilibili to watch the demo</a>；<a href="https://youtu.be/yMU-RWl55hg">Go to YouTube to watch the demo</a></p>

## Terminal Interaction Mode

<p>It is recommended to manage accounts through the configuration file. For more details, please refer to the <a href="https://github.com/JoeanAmier/TikTokDownloader/wiki/Documentation">Documentation</a></p>

![Terminal Mode Screenshot](docs/screenshot/Terminal Interaction Mode Screenshot CN1.png)
*****
![Terminal Mode Screenshot](docs/screenshot/Terminal Interaction Mode Screenshot CN2.png)
*****
![Terminal Mode Screenshot](docs/screenshot/Terminal Interaction Mode Screenshot CN3.png)

## Web UI Interaction Mode

> **The project code has been refactored, and the code for this mode has not been updated yet. It will be reopened when development is complete in the future!**

## Web API Interface Mode

![WebAPI Mode Screenshot](docs/screenshot/WebAPI Mode Screenshot CN1.png)
*****
![WebAPI Mode Screenshot](docs/screenshot/WebAPI Mode Screenshot CN2.png)

> **After starting this mode, access `http://127.0.0.1:5555/docs` or `http://127.0.0.1:5555/redoc` to view the automatically generated documentation!**

### API Example

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

# 📋 Project Description

## Quick Start

<p>⭐ Mac OS and Windows 10 and above users can go to <a href="https://github.com/JoeanAmier/TikTokDownloader/releases/latest">Releases</a> or <a href="https://github.com/JoeanAmier/TikTokDownloader/actions">Actions</a> to download the compiled program, and use it immediately!</p>
<p>⭐ This project includes GitHub Actions that automatically builds executable files, and users can use GitHub Actions to build executable files with the latest source code at any time!</p>
<p>⭐ Please refer to the <code>Guide to Build Executable Files</code> section of this document for tutorials on automatically building executable files; if you need more detailed graphic tutorials, please <a href="https://mp.weixin.qq.com/s/TorfoZKkf4-x8IBNLImNuw">refer to the article</a>!</p>
<p><strong>Note: The executable file <code>main</code> of the Mac OS platform may need to be started from the terminal command line; due to device limitations, the executable file of the Mac OS platform has not been tested, and its availability cannot be guaranteed!</strong></p>
<hr>
<ol>
<li><b>Run the executable file</b> or <b>Configure the environment to run</b>
<ol><b>Run the executable file</b>
<li>Download the executable file compressed package from <a href="https://github.com/JoeanAmier/TikTokDownloader/releases/latest">Releases</a> or Actions</li>
<li>Unzip and open the program folder, double-click to run <code>main</code></li>
</ol>
<ol><b>Configure the environment to run</b>

[//]: # (<li>Install the <a href="https://www.python.org/">Python</a> interpreter with version no less than <code>3.12</code></li>)
<li>Install the <a href="https://www.python.org/">Python</a> interpreter with version <code>3.12</code></li>
<li>Download the latest source code or source code released from <a href="https://github.com/JoeanAmier/TikTokDownloader/releases/latest">Releases</a> to your local machine</li>
<li>Run the command <code>python -m venv venv</code> to create a virtual environment (optional)</li>
<li>Run the command <code>.\venv\Scripts\activate.ps1</code> or <code>venv\Scripts\activate</code> to activate the virtual environment (optional)</li>
<li>Run the command <code>pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt</code> to install the required modules for the program</li>
<li>Run the command <code>python .\main.py</code> or <code>python main.py</code> to start DouK-Downloader</li>
</ol>
</li>
<li>Read the disclaimer of DouK-Downloader and enter the content according to the prompts</li>
<li>Write Cookie information to the configuration file
<ol><b>Read Cookie from clipboard</b>
<li>Refer to <a href="https://github.com/JoeanAmier/TikTokDownloader/blob/master/docs/Cookie%E8%8E%B7%E5%8F%96%E6%95%99%E7%A8%8B.md">Cookie extraction tutorial</a>, copy the required Cookie to the clipboard</li>
<li>Select the <code>Read Cookie from clipboard</code> option, and the program will automatically read the Cookie from the clipboard and write it to the configuration file</li>
</ol>
<ol><b>Read Cookie from browser</b>
<li>Select the <code>Read Cookie from browser</code> option, and enter the browser type or serial number according to the prompts</li>
</ol>
<ol><b><del>Scan code login to get Cookie</del> (invalid)</b>
<li><del>Select the <code>Scan code login to get Cookie</code> option, the program will display the login QR code image, and open the image with the default application</del></li>
<li><del>Use the Douyin APP to scan the QR code and log in to the account</del></li>
<li><del>Follow the prompts to operate, the program will automatically write the Cookie to the configuration file</del></li>
</ol>
</li>
<li>Return to the program interface and select <code>Terminal Interaction Mode</code> -> <code>Batch download link works (general)</code> -> <code>Manually enter the works links to be collected</code></li>
<li>Enter the Douyin work link to download the work file (the TikTok platform requires more initial settings, see the documentation for details)</li>
<li>For more detailed instructions, please see <b><a href="https://github.com/JoeanAmier/TikTokDownloader/wiki/Documentation">Project Documentation</a></b></li>
</ol>
<p>⭐ It is recommended to use <a href="https://learn.microsoft.com/zh-cn/windows/terminal/install">Windows Terminal</a> (Windows 11 comes with a default terminal)</p>

### Docker Container

<ol>
<li>Get the image</li>
<ul>
<li>Method 1: Use the <code>Dockerfile</code> file to build the image</li>
<li>Method 2: Use the command <code>docker pull joeanamier/tiktok-downloader</code> to pull the image</li>
<li>Method 3: Use the command <code>docker pull ghcr.io/joeanamier/tiktok-downloader</code> to pull the image</li>
</ul>
<li>Create a container: <code>docker run --name container name(optional) -p host port number:5555 -v tiktok_downloader_volume:/app/Volume -it &lt;image name&gt;</code>
</li>
<br><b>Note:</b> The <code>&lt;image name&gt;</code> here needs to be consistent with the image name you used in the first step (e.g. <code>joeanamier/tiktok-downloader</code> or <code>ghcr.io/joeanamier/tiktok-downloader</code>)
<li>Run the container
<ul>
<li>Start the container: <code>docker start -i container name/container ID</code></li>
<li>Restart the container: <code>docker restart -i container name/container ID</code></li>
</ul>
</li>
</ol>
<p>Docker containers cannot directly access the file system of the host machine, and some functions are unavailable, such as: <code>Read Cookie from browser</code>; if other functions have exceptions, please provide feedback!</p>
<hr>

## About Cookie

[Click to view the Cookie tutorial](https://github.com/JoeanAmier/TikTokDownloader/blob/master/docs/Cookie%E8%8E%B7%E5%8F%96%E6%95%99%E7%A8%8B.md)

> *  Cookies only need to be rewritten in the configuration file after they expire, and do not need to be written to the configuration file every time the program is run!
>
> *  Cookies will affect the resolution of the downloaded video files. If you cannot download the highest resolution video files, try updating the Cookie!
>
> *  When the program fails to get data, you can try to update the Cookie or use the Cookie that has been logged in!

<hr>

## Other Instructions

<ul>
<li>When the program prompts the user to enter, pressing Enter directly means returning to the previous menu, and entering <code>Q</code> or <code>q</code> means ending the run</li>
<li>Since the data for obtaining liked works and collected works only returns the publishing date of the liked/collected works, and does not return the operation date, the program needs to obtain all the data of the liked/collected works before date filtering; if the number of works is large, it may take a long time; the <code>max_pages</code> parameter can be used to control the number of requests</li>
<li>Obtaining the published work data of private accounts requires the Cookie after logging in, and the logged-in account needs to follow the private account</li>
<li>When batch downloading account works or collection works, if the corresponding nickname or identifier changes, the program will automatically update the nickname and identifier in the downloaded work file name</li>
<li>When the program downloads files, it will first download the files to a temporary folder, and then move them to the storage folder after the download is complete; the temporary folder will be emptied when the program finishes running</li>
<li>The <code>Batch download collection works mode</code> currently only supports downloading the collection works of the account corresponding to the currently logged-in Cookie, and does not support multiple accounts</li>
<li>If you want the program to use a proxy to request data, you must set the <code>proxy</code> parameter in <code>settings.json</code>, otherwise the program will not use a proxy</li>
<li>If your computer does not have a suitable program to edit JSON files, it is recommended to use the <a href="https://try8.cn/tool/format/json">JSON online tool</a> to edit the configuration file content</li>
<li>When the program requests the user to enter content or a link, please avoid the content or link entered containing line breaks, which may cause unexpected problems</li>
<li>This project will not support the download of paid works, please do not provide any issues about the download of paid works</li>
<li>Windows systems need to run the program as an administrator to read the Cookie of Chromium, Chrome, and Edge browsers</li>
<li>This project has not been optimized for the situation of multiple program openings. If you need multiple program openings, please copy the entire project folder to avoid unexpected problems</li>
<li>During the running of the program, if you need to terminate the program or <code>ffmpeg</code>, please press <code>Ctrl + C</code> to terminate the running, and do not directly click the close button of the terminal window</li>
</ul>
<h2>Guide to Build Executable Files</h2>
<details>
<summary><b>Guide to Build Executable Files (click to expand)</b></summary>

This guide will guide you to automatically complete the building and packaging of the program based on the latest source code by forking this repository and executing GitHub Actions!

---

### Steps to use

#### 1. Fork this repository

1. Click the **Fork** button in the upper right corner of the project repository to fork this repository to your personal GitHub account
2. Your Fork repository address will be similar to: `https://github.com/your-username/this-repo`

---

#### 2. Enable GitHub Actions

1. Go to the page of the repository you forked
2. Click the **Settings** tab at the top
3. Click the **Actions** tab on the right
4. Click the **General** option
5. Under **Actions permissions**, select the **Allow all actions and reusable workflows** option, and click the **Save** button

---

#### 3. Manually trigger the packaging process

1. In the repository you forked, click the **Actions** tab at the top
2. Find the workflow named **Build executable files**
3. Click the **Run workflow** button on the right:
    - Select the **master** or **develop** branch
    - Click **Run workflow**

---

#### 4. View the packaging progress

1. On the **Actions** page, you can see the running record of the triggered workflow
2. Click the running record and view the detailed logs to understand the packaging progress and status

---

#### 5. Download the packaging results

1. After the packaging is complete, go to the corresponding running record page
2. In the **Artifacts** section at the bottom of the page, you will see the packaged result files
3. Click to download and save locally, and you can get the packaged program

---

### Notes

1. **Resource usage**:
    - The Actions running environment is provided free of charge by GitHub, and ordinary users have a certain amount of free usage per month (2000 minutes)

2. **Code modification**:
    - You can freely modify the code in the Fork repository to customize the program packaging process
    - After modification, re-trigger the packaging process, you will get a customized build version

3. **Keep in sync with the main repository**:
    - If the main repository updates the code or workflow, it is recommended that you regularly synchronize the Fork repository to get the latest features and fixes

---

### Actions FAQ

#### Q1: Why can't I trigger the workflow?

A: Please confirm that you have followed the steps to **enable Actions**, otherwise GitHub will prohibit running the workflow

#### Q2: What should I do if the packaging process fails?

A:

- Check the running logs to understand the cause of the failure
- Make sure the code has no syntax errors or dependency issues
- If the problem is still not resolved, you can raise the issue on the [Issues page](https://github.com/JoeanAmier/TikTokDownloader/issues) of this repository

#### Q3: Can I directly use the Actions of the main repository?

A: Due to permission restrictions, you cannot directly trigger the Actions of the main repository. Please perform the packaging process by forking the repository

</details>

## Program Update

<p><strong>Option 1:</strong> Download and unzip the file, and copy the old version of the <code>_internal\Volume</code> folder to the new version of the <code>_internal</code> folder.</p>
<p><strong>Option 2:</strong> Download and unzip the file (do not run the program), copy all the files, and directly overwrite the old version files.</p>

# ⚠️ Disclaimer

<ol>
<li>The use of this project by the user is decided by the user, and the user shall bear the risk. The author is not responsible for any loss, liability, or risk arising from the user's use of this project.</li>
<li>The code and functions provided by the author of this project are based on the development results of existing knowledge and technology. The author strives to ensure the correctness and security of the code at the current technical level, but does not guarantee that the code is completely free of errors or defects.</li>
<li>All third-party libraries, plugins or services on which this project depends, each follow their original open source or commercial licenses. The user needs to check and abide by the corresponding agreements. The author does not assume any responsibility for the stability, security and compliance of the third-party components.</li>
<li>When using this project, users must strictly abide by the requirements of <a href="https://github.com/JoeanAmier/TikTokDownloader/blob/master/LICENSE">GNU
    General Public License v3.0</a>, and indicate the use of <a
        href="https://github.com/JoeanAmier/TikTokDownloader/blob/master/LICENSE">GNU General Public License
    v3.0</a> code in the appropriate place.
</li>
<li>When users use the code and functions of this project, they must study the relevant laws and regulations by themselves, and ensure that their use behavior is legal and compliant. Any legal liabilities and risks caused by violating laws and regulations shall be borne by the user.</li>
<li>Users shall not use this tool to engage in any acts that infringe intellectual property rights, including but not limited to unauthorized downloading and dissemination of copyrighted content. The developer does not participate in, support, or recognize the acquisition or distribution of any illegal content.</li>
<li>This project is not responsible for the compliance of the user's data collection, storage, transmission, and other processing activities. Users should comply with relevant laws and regulations to ensure that the processing behavior is legal and proper; the legal liability caused by violations shall be borne by the user.</li>
<li>Under no circumstances shall the author, contributors, or other relevant parties of this project be connected to the user's use behavior, or be required to be responsible for any loss or damage arising from the user's use of this project.</li>
<li>The author of this project will not provide a paid version of the DouK-Downloader project, nor will it provide any commercial services related to the DouK-Downloader project.</li>
<li>Any secondary development, modification, or compilation of the program based on this project has nothing to do with the original author, and the original author does not assume any responsibility related to the secondary development behavior or its results. Users shall be fully responsible for all situations that may be caused by secondary development.</li>
<li>This project does not grant any patent license to users; if the use of this project leads to patent disputes or infringement, the user shall bear all risks and responsibilities. Without the written authorization of the author or the rights holder, it is not allowed to use this project for any commercial publicity, promotion or re-authorization.</li>
<li>The author reserves the right to terminate the service to any user who violates this statement at any time, and may require them to destroy the code and derivative works that have been obtained.</li>
<li>The author reserves the right to update this statement without prior notice, and the user's continued use is deemed to accept the revised terms.</li>
</ol>
<b>Before using the code and functions of this project, please consider carefully and accept the above disclaimer. If you have any questions or disagree with the above statement, please do not use the code and functions of this project. If you use the code and functions of this project, it is deemed that you have fully understood and accepted the above disclaimer, and voluntarily bear all the risks and consequences of using this project.</b>
<h1>🌟 Contribution Guide</h1>
<p><strong>Welcome to contribute to this project! In order to keep the code library clean, efficient and easy to maintain, please read the following guidelines carefully to ensure that your contributions can be smoothly accepted and integrated.</strong></p>
<ul>
<li>Before starting development, please pull the latest code from the <code>develop</code> branch and use it as the basis for modification; this helps avoid merge conflicts and ensures that your changes are based on the latest project status.</li>
<li>If your changes involve multiple unrelated functions or issues, please divide them into multiple independent commits or pull requests.</li>
<li>Each pull request should focus on a single function or fix as much as possible to facilitate code review and testing.</li>
<li>Follow the existing code style; please ensure that your code is consistent with the existing code style in the project; it is recommended to use the Ruff tool to maintain the code format specification.</li>
<li>Write readable code; add appropriate comments to help others understand your intentions.</li>
<li>Each submission should include a clear and concise submission message to describe the changes made. The submission message should follow the following format: <code>&lt;type&gt;: &lt;brief description&gt;</code></li>
<li>When you are ready to submit a pull request, please give priority to submitting them to the <code>develop</code> branch; this is to give the maintainer a buffer, and perform additional testing and review before finally merging into the <code>master</code>
branch.</li>
<li>It is recommended to communicate with the author before development or when you have questions to ensure that the development direction is consistent and avoid duplication of work or invalid submissions.</li>
</ul>
<p><strong>References:</strong></p>
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
<p>If you are willing, you can consider providing funding to provide additional support for <b>DouK-Downloader</b>!</p>

# 💰 Project Sponsorship

## DartNode

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")

***

## ZMTO

<a href="https://www.zmto.com/"><img src="https://console.zmto.com/templates/2019/dist/images/logo_dark.svg" alt="ZMTO"></a>
<p><a href="https://www.zmto.com/">ZMTO</a>: A professional cloud infrastructure provider, providing efficient solutions with reliable cutting-edge technology and professional support, and providing enterprise-level VPS infrastructure for eligible open source projects, supporting the sustainable development and innovation of the open source ecosystem.</p>

***

## TikHub

<p><a href="https://tikhub.io/">TikHub</a>: A third-party API service provider that focuses on providing APIs for various platforms.</p>
<p>Through daily check-in, users can get a small amount of usage quota for free; you can use my <strong>Referral Link</strong>: <a href="https://user.tikhub.io/users/signup?referral_code=ZrdH8McC">https://user.tikhub.io/users/signup?referral_code=ZrdH8McC</a> or <strong>Referral Code</strong>: <code>ZrdH8McC</code>, register and recharge to get <code>$2</code> quota!</p>

# ✉️ Contact the Author

<ul>
<li>Author's email: yonglelolu@foxmail.com</li>
<li>Author's WeChat: Downloader_Tools</li>
<li>WeChat Official Account: Downloader Tools</li>
<li><b>Discord Community</b>: <a href="https://discord.com/invite/ZYtmgKud9Y">Click to join the community</a></li>
<li>QQ Group Chat (project communication): <a href="https://github.com/JoeanAmier/TikTokDownloader/blob/master/docs/QQ%E7%BE%A4%E8%81%8A%E4%BA%8C%E7%BB%B4%E7%A0%81.png">Scan code to join the group chat</a></li>
</ul>
<p>✨ <b>The author's other open source projects:</b></p>
<ul>
<li><b>XHS-Downloader (Xiaohongshu, RedNote)</b>: <a href="https://github.com/JoeanAmier/XHS-Downloader">https://github.com/JoeanAmier/XHS-Downloader</a></li>
<li><b>KS-Downloader (Kuaishou)</b>: <a href="https://github.com/JoeanAmier/KS-Downloader">https://github.com/JoeanAmier/KS-Downloader</a></li>
</ul>
<h1>⭐ Star Trend</h1>
<p>
<img alt="Star History Chart" src="https://api.star-history.com/svg?repos=JoeanAmier/TikTokDownloader&amp;type=Timeline"/>
</p>

# 💡 Project References

* https://github.com/Johnserf-Seed/f2
* https://github.com/Johnserf-Seed/TikTokDownload
* https://github.com/Evil0ctal/Douyin_TikTok_Download_API
* https://github.com/NearHuiwen/TiktokDouyinCrawler
* https://github.com/ihmily/DouyinLiveRecorder
* https://github.com/encode/httpx/
* https://github.com/Textualize/rich
* https://github.com/omnilib/aiosqlite
* https://github.com/Tinche/aiofiles
* https://github.com/thewh1teagle/rookie
* https://github.com/pyinstaller/pyinstaller
* https://foss.heptapod.net/openpyxl/openpyxl
* https://github.com/carpedm20/emoji/
* https://github.com/lxml/lxml
* https://ffmpeg.org/ffmpeg-all.html
* https://www.tikwm.com/