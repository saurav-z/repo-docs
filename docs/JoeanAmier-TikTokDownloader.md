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

##  Download TikTok and Douyin Videos Effortlessly with DouK-Downloader!

DouK-Downloader is a powerful, open-source Python tool for downloading videos, images, and more from TikTok and Douyin. [Check out the original repository](https://github.com/JoeanAmier/TikTokDownloader)

<hr>

## Key Features

*   ✅ **Video & Image Downloads**:
    *   Download Douyin (抖音) and TikTok videos, including those without watermarks.
    *   Download high-quality video files.
    *   Download image sets (图集).
*   ✅ **Account & Content Downloads**:
    *   Download videos from Douyin and TikTok accounts (发布/喜欢).
    *   Download content from favorites/collections (收藏/收藏夹).
    *   Batch download of videos from collections and albums (合集).
*   ✅ **Live Stream Features**:
    *   Get and download live stream URLs from Douyin and TikTok.
    *   Download live stream videos.
*   ✅ **Data & Information Gathering**:
    *   Collect detailed account data from both platforms.
    *   Gather comments data of videos.
    *   Extract search results and trending data from Douyin.
*   ✅ **Flexible Usage**:
    *   Web UI and Web API are available.
    *   Supports CSV/XLSX/SQLite data saving.
    *   Supports proxy use for data collection.
    *   Multi-threading for faster downloads.
    *   Supports downloading using links from the clipboard.
*   ✅ **Additional Features**:
    *   Download dynamic/static cover images.
    *   Automatic skip of already downloaded files.
    *   Incremental account download support.
    *   File integrity checks.
    *   Custom file naming and saving.
    *   Supports Docker.

<hr>

## Program Screenshots

*   See the terminal interface in action:

    ![终端模式截图](docs/screenshot/终端交互模式截图CN1.png)
    *****
    ![终端模式截图](docs/screenshot/终端交互模式截图CN2.png)
    *****
    ![终端模式截图](docs/screenshot/终端交互模式截图CN3.png)

*   Web API Example

    ![WebAPI模式截图](docs/screenshot/WebAPI模式截图CN1.png)
    *****
    ![WebAPI模式截图](docs/screenshot/WebAPI模式截图CN2.png)

    *   Access the automatically generated API documentation at: `http://127.0.0.1:5555/docs` or `http://127.0.0.1:5555/redoc`

## Getting Started

1.  **Download and Run**:
    *   Download pre-built executables from the [Releases](https://github.com/JoeanAmier/TikTokDownloader/releases/latest) or [Actions](https://github.com/JoeanAmier/TikTokDownloader/actions) sections.

2.  **Configure Your Environment (Alternative):**
    *   Install Python 3.12.
    *   Clone or download the source code.
    *   Create and activate a virtual environment (optional): `python -m venv venv` & `.\venv\Scripts\activate` or `venv\Scripts\activate`.
    *   Install dependencies: `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`.
    *   Run the program: `python .\main.py` or `python main.py`.

3.  **Accept the Disclaimer**.

4.  **Set up Cookie**:
    *   **From Clipboard:** Follow the [Cookie extraction tutorial](https://github.com/JoeanAmier/TikTokDownloader/blob/master/docs/Cookie%E8%8E%B7%E5%8F%96%E6%95%99%E7%A8%8B.md).

5.  **Start Downloading**:
    *   In terminal mode, choose "批量下载链接作品(通用)" -> "手动输入待采集的作品链接".
    *   Enter a Douyin/TikTok video link to download.

6.  **For detailed information**: Please refer to the [Project Documentation](https://github.com/JoeanAmier/TikTokDownloader/wiki/Documentation).

### Docker Container

1.  **Get the Image**:
    *   Build from the `Dockerfile`.
    *   Pull from Docker Hub: `docker pull joeanamier/tiktok-downloader`
    *   Pull from GitHub Container Registry: `docker pull ghcr.io/joeanamier/tiktok-downloader`

2.  **Create the Container**: `docker run --name <container_name> -p host_port:5555 -v tiktok_downloader_volume:/app/Volume -it <image_name>`

3.  **Run the Container**:
    *   Start: `docker start -i <container_name/container_ID>`
    *   Restart: `docker restart -i <container_name/container_ID>`

<hr>

##  Cookie Information

*   [Cookie Extraction Tutorial](https://github.com/JoeanAmier/TikTokDownloader/blob/master/docs/Cookie%E8%8E%B7%E5%8F%96%E6%95%99%E7%A8%8B.md)

> *   Cookies only need to be updated if they expire; it isn't required to configure every time the program runs!
> *   Cookies can impact video resolution; if you are unable to download the highest resolution, try updating your cookie.

<hr>

## Other Notes

*   Type `Enter` to go back, and `Q` or `q` to quit.
*   When downloading account likes/favorites, the program may take more time because it needs to acquire the complete list.
*   Downloading private account content needs a login cookie and requires that the logged-in account follows the private account.
*   When downloading account or collection content, the program will automatically update the names and IDs in the downloaded files.
*   Files are temporarily downloaded, then moved to storage. The temporary folder is cleared upon exit.
*   When using proxies, `proxy` needs to be set up in `settings.json`.
*   Edit config files using an online JSON tool (e.g. https://try8.cn/tool/format/json).
*   Avoid line breaks in input links/content.
*   This project will not support paid video downloads.
*   Requires administrator rights to read browser cookies in Windows.
*   For multi-instance operation, copy the project folder.
*   Use `Ctrl + C` to end the process.
*   Check Disclaimer before using the code.

## Building Executable File

<details>
<summary><b>Build Executable File Guide (Click to Expand)</b></summary>

This guide leads you through forking the repository and executing GitHub Actions to automate the building and packaging of your program from the latest source code!

---

### Steps

#### 1. Fork the Repository

1.  Click the **Fork** button in the top-right of the project's repository.
2.  Your forked repository will look similar to: `https://github.com/your-username/this-repo`

---

#### 2. Activate GitHub Actions

1.  Navigate to your forked repository page.
2.  Click on the **Settings** tab at the top.
3.  Click on **Actions** in the right-hand sidebar.
4.  Click the **General** option.
5.  Under **Actions permissions**, choose the **Allow all actions and reusable workflows** option, and click **Save**.

---

#### 3. Manually Trigger the Build Workflow

1.  In your forked repository, click the **Actions** tab at the top.
2.  Find the workflow named **构建可执行文件**.
3.  Click the **Run workflow** button on the right:
    -   Choose the branch: either **master** or **develop**.
    -   Click **Run workflow**.

---

#### 4. Monitor the Build Progress

1.  On the **Actions** page, view the workflow run history.
2.  Click the run record to view the detailed logs and the build progress.

---

#### 5. Download the Build Artifacts

1.  After the build completes, go to the run history page.
2.  Find the built files in the **Artifacts** section at the bottom of the page.
3.  Click to download and save the artifacts to your local machine to acquire the built program.

---

### Notes

1.  **Resource Usage**:
    -   The Actions environment is provided for free by GitHub, but standard users have a limited free usage allowance each month (2000 minutes).

2.  **Code Modifications**:
    -   You can freely modify the code in your forked repository to customize the program's build process.
    -   Trigger the build workflow again after any changes, and you'll receive a customized build version.

3.  **Keep in Sync with the Main Repository**:
    -   It's recommended to sync your forked repository regularly with the main repository to obtain the latest features and fixes.

---

### Common Actions Issues

#### Q1: Why can't I trigger the workflow?

A: Confirm that you have followed the **Activate Actions** steps; otherwise, GitHub will block the execution of the workflow.

#### Q2: What if the build process fails?

A:

-   Check the logs to see why it failed.
-   Ensure that the code has no syntax errors or dependency problems.
-   If the problem is unresolved, you can submit an issue on the [Issues page](https://github.com/JoeanAmier/TikTokDownloader/issues) of this repository.

#### Q3: Can I directly utilize the main repository's Actions?

A: You cannot directly trigger the main repository's Actions due to permission restrictions. Forking the repository is necessary to execute the build process.

</details>

## Updating the Program

*   **Method 1**: Copy the `_internal\Volume` folder from the old version into the new version's `_internal` folder.
*   **Method 2**: Download and extract the new version (do not run it), copy all files, and overwrite the old version's files.

<hr>

## ⚠️ Disclaimer

*   The user is fully responsible for using this project and accepts the associated risks. The author disclaims liability for any loss, damage, or risk resulting from the user's use of this project.
*   The author provides code and functions based on current knowledge and technology. While the author strives to ensure the correctness and security of the code, the code is not guaranteed to be entirely free of errors or defects.
*   Any third-party libraries, plugins, or services used by this project are subject to their respective open-source or commercial licenses, which users are expected to review and abide by. The author assumes no responsibility for the stability, security, or compliance of third-party components.
*   Users must adhere to the requirements of the [GNU General Public License v3.0](https://github.com/JoeanAmier/TikTokDownloader/blob/master/LICENSE) and acknowledge the use of code under this license in the appropriate places.
*   Users must research and comply with relevant laws and regulations when using the project's code and functionality, and ensure that their usage is lawful and compliant. The user is responsible for any legal liability and risks arising from violations of laws and regulations.
*   Users may not use this tool for any activities that infringe intellectual property rights, including but not limited to unauthorized downloading and distribution of copyrighted content. The developer does not participate in, support, or endorse the acquisition or distribution of illegal content.
*   The project is not responsible for the compliance of the user's data collection, storage, transmission, and other processing activities. Users must comply with relevant laws and regulations and ensure that processing activities are lawful. The user is responsible for any legal liability resulting from illegal operations.
*   Users may not associate the author, contributors, or other related parties of this project with their use or ask them to be responsible for any loss or damage resulting from the user's use of the project.
*   The author will not provide a paid version or commercial services related to this project.
*   Any secondary development, modification, or compilation based on this project is unrelated to the original author. The original author is not responsible for any results or situations arising from secondary development. Users should assume full responsibility for all risks and consequences.
*   This project does not grant any patent licenses. If using this project leads to patent disputes or infringement, the user assumes all risks and responsibilities. The user may not use this project for any commercial promotion or re-authorization without the written authorization of the author or the rights holder.
*   The author reserves the right to terminate service to any user who violates this statement and may require the destruction of the acquired code and derivative works.
*   The author reserves the right to update this statement without notice, and continued use by the user constitutes acceptance of the revised terms.

**Please carefully consider and accept the disclaimer before using this project's code and functions. If you have any questions or disagree with the above statement, please do not use the code and functions of this project. If you use the code and functions of this project, it is considered that you have fully understood and accepted the above disclaimer and voluntarily assume all risks and consequences of using this project.**

<hr>

<h1>🌟 Contribution Guide</h1>

**Contributions are welcome! To ensure your contribution is smoothly accepted and integrated, carefully review the following guidelines.**

*   Before starting development, pull the latest code from the `develop` branch and use it as a base for your changes. This helps avoid merge conflicts and ensures that your changes are based on the latest project state.
*   If your changes involve multiple unrelated features or issues, separate them into multiple independent commits or pull requests.
*   Each pull request should focus on a single feature or fix as much as possible to facilitate code review and testing.
*   Follow the existing code style. Ensure your code aligns with the project's existing code style. It is recommended to use the Ruff tool to maintain code format compliance.
*   Write readable code; add appropriate comments to help others understand your intentions.
*   Each commit should include a clear and concise commit message describing the changes. Commit messages should follow the format: `<type>: <short description>`.
*   When ready to submit a pull request, prioritize submitting them to the `develop` branch. This provides a buffer for the maintainer to conduct additional testing and review before the final merge into the `master` branch.
*   It is recommended to communicate with the author before or when encountering issues to ensure consistency in the development direction and avoid redundant effort or invalid submissions.

**Resources**:

*   [Contributor Covenant](https://www.contributor-covenant.org/zh-cn/version/2/1/code_of_conduct/)
*   [How to contribute to open source](https://opensource.guide/zh-hans/how-to-contribute/)

<hr>

# ♥️ Support the Project

If **DouK-Downloader** is helpful to you, consider giving it a **Star** ⭐. Thank you for your support!

<table>
<thead>
<tr>
<th align="center">WeChat</th>
<th align="center">Alipay</th>
</tr>
</thead>
<tbody><tr>
<td align="center"><img src="./docs/微信赞助二维码.png" alt="WeChat Donation QR Code" height="200" width="200"></td>
<td align="center"><img src="./docs/支付宝赞助二维码.png" alt="Alipay Donation QR Code" height="200" width="200"></td>
</tr>
</tbody>
</table>

You can also consider providing funding to support **DouK-Downloader**!

<hr>

# 💰 Project Sponsorship

## DartNode

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")

***

## ZMTO

<a href="https://www.zmto.com/"><img src="https://console.zmto.com/templates/2019/dist/images/logo_dark.svg" alt="ZMTO"></a>
<p><a href="https://www.zmto.com/">ZMTO</a>: A professional cloud infrastructure provider offers efficient solutions with reliable, cutting-edge technology and expert support and provides enterprise-grade VPS infrastructure for eligible open-source projects to support the sustainable development and innovation of the open-source ecosystem.</p>

***

## TikHub

<p><a href="https://tikhub.io/">TikHub</a>: A third-party API service provider specializing in providing APIs for various platforms.</p>
<p>By daily check-in, users can get a small amount of usage quota for free; you can use my <strong>referral link</strong>: <a href="https://user.tikhub.io/users/signup?referral_code=ZrdH8McC">https://user.tikhub.io/users/signup?referral_code=ZrdH8McC</a> or the <strong>referral code</strong>: `ZrdH8McC` to sign up and get a credit of `$2`!</p>

<hr>

# ✉️ Contact

*   Author Email: yonglelolu@foxmail.com
*   Author WeChat: Downloader_Tools
*   WeChat Official Account: Downloader Tools
*   <b>Discord Community</b>: <a href="https://discord.com/invite/ZYtmgKud9Y">Join the Community</a>
*   QQ Group Chat (Project Exchange): <a href="https://github.com/JoeanAmier/TikTokDownloader/blob/master/docs/QQ%E7%BE%A4%E8%81%8A%E4%BA%8C%E7%BB%B4%E7%A0%81.png">Scan to Join</a>

<p>✨ <b>Other Open Source Projects by the Author:</b></p>

*   <b>XHS-Downloader (小红书、XiaoHongShu、RedNote)</b>: <a href="https://github.com/JoeanAmier/XHS-Downloader">https://github.com/JoeanAmier/XHS-Downloader</a>
*   <b>KS-Downloader (快手、KuaiShou)</b>: <a href="https://github.com/JoeanAmier/KS-Downloader">https://github.com/JoeanAmier/KS-Downloader</a>

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