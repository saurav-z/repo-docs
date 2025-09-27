<div align="center">
<img src="./static/images/DouK-Downloader.png" alt="DouK-Downloader" height="256" width="256"><br>
<h1>DouK-Downloader: The Ultimate TikTok and Douyin Video Downloader</h1>
<p>Download, save, and enjoy your favorite TikTok and Douyin content effortlessly with this powerful and open-source tool.</p>
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

<p>🔥 <b>DouK-Downloader</b> is a free, open-source tool built on HTTPX for effortlessly downloading TikTok and Douyin content. Capture data and download videos, images, music, and more from TikTok and Douyin. Supports batch downloading, live streams, comments, and account data acquisition. (Project formerly known as <code>TikTokDownloader</code>) </p>

<hr>

## Key Features

*   ✅ **Download High-Quality Videos:** Download TikTok and Douyin videos without watermarks.
*   ✅ **Account & Content Downloading:** Batch download videos from accounts, including published, liked, and collection content.
*   ✅ **Live Stream Downloads:** Capture live streams from both platforms.
*   ✅ **Data Acquisition:** Gather detailed account and content data, including comments and trending topics.
*   ✅ **Versatile Output:** Supports saving data in CSV, XLSX, and SQLite formats.
*   ✅ **Web API Integration:** Access functionalities through a Web API.
*   ✅ **Cross-Platform Compatibility:** Works on Windows, macOS, and Docker.

## Getting Started

### Installation

1.  **Choose Your Method:**

    *   **Pre-built Executable:** Download pre-compiled executables from the [Releases](https://github.com/JoeanAmier/TikTokDownloader/releases/latest) or [Actions](https://github.com/JoeanAmier/TikTokDownloader/actions) sections.
    *   **Install from Source (Requires Python 3.12):**
        1.  Install Python 3.12 from [Python's Website](https://www.python.org/).
        2.  Clone or download the repository.
        3.  Create a virtual environment: `python -m venv venv` (Optional).
        4.  Activate the virtual environment: `.\venv\Scripts\activate.ps1` (Windows) or `source venv/Scripts/activate` (macOS/Linux) (Optional).
        5.  Install dependencies: `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`.
        6.  Run the application: `python main.py` or `python .\main.py`.
2.  **Read and Accept Disclaimer:** Acknowledge the disclaimer presented by the application.
3.  **Configure Cookie:** Import your TikTok/Douyin cookies to enable certain features:
    *   **From Clipboard:** Copy the cookie to your clipboard, and select the appropriate option in the app.
    *   **From Browser:** Select the option to read cookies from your browser, providing browser type or index as prompted.

4.  **Start Downloading:** Select your desired mode (Terminal, Web UI (coming soon), or Web API) and enter video links or account details.
5.  **Documentation:** For detailed information and guides, consult the comprehensive [Project Documentation](https://github.com/JoeanAmier/TikTokDownloader/wiki/Documentation).

### Docker

1.  **Get the Image:** Use a prebuilt image or build your own from the Dockerfile.
    *   Pull a prebuilt image: `docker pull joeanamier/tiktok-downloader` or `docker pull ghcr.io/joeanamier/tiktok-downloader`
2.  **Create and Run a Container:**
    `docker run --name <container_name> -p 5555:5555 -v tiktok_downloader_volume:/app/Volume -it <image_name>`
3. **Start/Restart Container:**
    *   Start: `docker start -i <container_name/container_id>`
    *   Restart: `docker restart -i <container_name/container_id>`
**Important:** Docker containers may have limitations.

## About Cookies

[Click here to view the Cookie tutorial](https://github.com/JoeanAmier/TikTokDownloader/blob/master/docs/Cookie%E8%8E%B7%E5%8F%96%E6%95%99%E7%A8%8B.md)

>   *   Update cookies if your video resolutions are low or to resolve data retrieval issues.
>   *   Cookies are only required when they expire.

<hr>

## Additional Notes

*   Press Enter to return to the previous menu or enter `Q` or `q` to quit.
*   Cookie is needed to download content from private accounts.
*   Update file names for batch downloads if account names change.
*   Files are first saved to a temporary folder and moved to the destination folder after download.
*   Use proxies for data requests by setting up the `proxy` parameter in `settings.json`.
*   Use a JSON online tool to edit the configuration file.
*   Run the program as an administrator to access browser cookies on Windows.
*   Do not close the terminal window directly. Use `Ctrl + C` to terminate running programs.

<h2>Building Executable Files Guide</h2>
<details>
<summary><b>Build Executable Files Guide (Click to Expand)</b></summary>

This guide will take you through forking this repository and automatically building and packaging the application based on the latest source code by executing GitHub Actions!

---

### Steps

#### 1. Fork the Repository

1.  Click the **Fork** button in the top-right corner of the project repository to fork this repository to your personal GitHub account.
2.  Your forked repository's address will be similar to this: `https://github.com/your-username/this-repo`

---

#### 2. Enable GitHub Actions

1.  Go to your forked repository page.
2.  Click the **Settings** tab at the top.
3.  Click the **Actions** tab on the right.
4.  Click the **General** tab.
5.  Under the **Actions permissions** section, choose the **Allow all actions and reusable workflows** option, then click the **Save** button.

---

#### 3. Manually Trigger the Packaging Process

1.  In your forked repository, click the **Actions** tab at the top.
2.  Find the workflow named **Build Executable File**.
3.  Click the **Run workflow** button on the right:
    -   Select the **master** or **develop** branch.
    -   Click **Run workflow**.

---

#### 4. View the Packaging Progress

1.  In the **Actions** page, you can view the workflow execution record.
2.  Click the execution record to view the detailed logs and understand the packaging progress and status.

---

#### 5. Download the Packaging Results

1.  After packaging is complete, go to the corresponding execution record page.
2.  In the **Artifacts** section at the bottom of the page, you will see the packaged result files.
3.  Click to download and save them locally to get the packaged application.

---

### Notes

1.  **Resource Usage:**
    -   Actions' execution environment is provided free of charge by GitHub, and ordinary users have a certain amount of free usage quota per month (2000 minutes).

2.  **Code Modification:**
    -   You can freely modify the code in your forked repository to customize the application's packaging process.
    -   After modifying, trigger the packaging process again, and you will get the customized build version.

3.  **Keep in Sync with the Main Repository:**
    -   If the main repository updates the code or workflow, it is recommended that you periodically synchronize your forked repository to obtain the latest features and fixes.

---

### Actions FAQs

#### Q1: Why can't I trigger the workflow?

A: Please confirm that you have followed the **Enable Actions** step; otherwise, GitHub will prohibit the workflow from running.

#### Q2: What should I do if the packaging process fails?

A:

-   Check the execution logs to understand the cause of the failure.
-   Ensure that the code has no syntax errors or dependency issues.
-   If the problem persists, you can ask questions on the [Issues page](https://github.com/JoeanAmier/TikTokDownloader/issues) of this repository.

#### Q3: Can I directly use the main repository's Actions?

A: Due to permission restrictions, you cannot directly trigger the main repository's Actions. Please use the fork repository method to perform the packaging process.

</details>

## Updating the Program

**Option 1:** Copy the `_internal\Volume` folder from the old version into the new version's `_internal` folder after downloading and extracting.

**Option 2:** Download and extract (do not run the program). Copy all files, and replace the files in the old version directly.

# ⚠️ Disclaimer

<ol>
<li>Users are solely responsible for the use of this project and assume all associated risks. The author is not liable for any losses, responsibilities, or risks arising from the use of this project.</li>
<li>The code and functionalities provided by the author are based on existing knowledge and technical developments. The author strives to ensure the correctness and security of the code to the best of their technical ability but does not guarantee that the code is entirely free of errors or defects.</li>
<li>All third-party libraries, plugins, or services used in this project are subject to their original open-source or commercial licenses. Users should review and adhere to the respective agreements. The author is not responsible for the stability, security, or compliance of any third-party components.</li>
<li>Users must strictly comply with the requirements of <a href="https://github.com/JoeanAmier/TikTokDownloader/blob/master/LICENSE">GNU General Public License v3.0</a> when using this project and should indicate the use of the <a href="https://github.com/JoeanAmier/TikTokDownloader/blob/master/LICENSE">GNU General Public License v3.0</a> code in appropriate places.</li>
<li>Users must independently research relevant laws and regulations when using the code and functionalities of this project and ensure that their usage complies with all legal requirements. Users are solely responsible for any legal liabilities and risks arising from violations of laws and regulations.</li>
<li>Users must not use this tool for any activities that infringe on intellectual property rights, including but not limited to unauthorized downloading and distribution of copyrighted content. The developer does not participate in, support, or endorse the acquisition or distribution of any illegal content.</li>
<li>This project is not responsible for the compliance of user data collection, storage, transmission, and other processing activities. Users should independently comply with relevant laws and regulations to ensure that their processing activities are legal and proper. Users are solely responsible for any legal liabilities resulting from non-compliant operations.</li>
<li>Users shall not, under any circumstances, associate the author, contributors, or other related parties of this project with the user's usage behavior or request them to be responsible for any losses or damages incurred by the user's use of this project.</li>
<li>The author of this project will not provide a paid version of the DouK-Downloader project or offer any commercial services related to the DouK-Downloader project.</li>
<li>Any secondary development, modification, or compiled programs based on this project are not related to the original author, and the original author is not responsible for any liabilities related to secondary development activities or their results. Users are solely responsible for all situations that may arise from secondary development.</li>
<li>This project does not grant users any patent licenses. If using this project leads to patent disputes or infringements, users assume all risks and responsibilities. Without written authorization from the author or rights holder, users must not use this project for any commercial promotion, marketing, or sublicensing.</li>
<li>The author reserves the right to terminate providing services to any users who violate this statement and may require them to destroy acquired code and derivative works.</li>
<li>The author reserves the right to update this statement without prior notice, and continued use by users constitutes acceptance of the revised terms.</li>
</ol>
<b>Before using the code and functionalities of this project, please carefully consider and accept the above disclaimer. If you have any questions or do not agree with the above statements, please do not use the code and functionalities of this project. By using the code and functionalities of this project, you are deemed to have fully understood and accepted the above disclaimer and voluntarily assume all risks and consequences of using this project.</b>
<h1>🌟 Contribution Guidelines</h1>
<p><strong>Welcome to contribute to this project! Please review the following guidelines to ensure your contributions are smoothly integrated while maintaining a clean and efficient codebase.</strong></p>
<ul>
<li>Before starting development, pull the latest code from the <code>develop</code> branch to base your changes on; this helps to avoid merge conflicts and ensures your changes are based on the most current project status.</li>
<li>If your changes involve several unrelated features or issues, separate them into independent commits or pull requests.</li>
<li>Each pull request should focus on a single functionality or fix to facilitate code review and testing.</li>
<li>Follow the existing code style; ensure your code aligns with the existing code style within the project; using the Ruff tool is recommended to maintain code format standards.</li>
<li>Write readable code; add appropriate comments to help others understand your intentions.</li>
<li>Each commit should include a clear, concise commit message describing the changes made. Commit messages should follow the following format: <code>&lt;type&gt;: &lt;brief description&gt;</code></li>
<li>When preparing to submit a pull request, please submit them to the <code>develop</code> branch first; this serves as a buffer for the maintainers to perform additional testing and review before merging into the <code>master</code> branch.</li>
<li>It is recommended to communicate with the author before development or when encountering questions to ensure consistent development directions and avoid redundant work or invalid submissions.</li>
</ul>
<p><strong>References:</strong></p>
<ul>
<li><a href="https://www.contributor-covenant.org/zh-cn/version/2/1/code_of_conduct/">Contributor Covenant</a></li>
<li><a href="https://opensource.guide/zh-hans/how-to-contribute/">How to Contribute to Open Source</a></li>
</ul>

# ♥️ Support the Project

<p>If <b>DouK-Downloader</b> is useful to you, please consider giving it a <b>Star</b> ⭐, Thank you for your support!</p>
<table>
<thead>
<tr>
<th align="center">WeChat</th>
<th align="center">Alipay</th>
</tr>
</thead>
<tbody><tr>
<td align="center"><img src="./docs/微信赞助二维码.png" alt="WeChat Sponsor QR Code" height="200" width="200"></td>
<td align="center"><img src="./docs/支付宝赞助二维码.png" alt="Alipay Sponsor QR Code" height="200" width="200"></td>
</tr>
</tbody>
</table>
<p>If you wish, you can consider providing financial support to <b>DouK-Downloader</b>!</p>

# 💰 Project Sponsors

## DartNode

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")

***

## ZMTO

<a href="https://www.zmto.com/"><img src="https://console.zmto.com/templates/2019/dist/images/logo_dark.svg" alt="ZMTO"></a>
<p><a href="https://www.zmto.com/">ZMTO</a>: A professional cloud infrastructure provider, offering efficient solutions with reliable cutting-edge technology and professional support. They provide enterprise-level VPS infrastructure for eligible open-source projects, supporting the sustainable development and innovation of the open-source ecosystem.</p>

***

## TikHub

<p><a href="https://tikhub.io/">TikHub</a>: A third-party API service provider, specializing in providing APIs for various platforms.</p>
<p>By signing in daily, users can obtain a small amount of free usage; You can use my <strong>referral link</strong>: <a href="https://user.tikhub.io/users/signup?referral_code=ZrdH8McC">https://user.tikhub.io/users/signup?referral_code=ZrdH8McC</a> or the <strong>referral code</strong>: <code>ZrdH8McC</code>, to register and top up to get <code>$2</code> worth of credit!</p>

# ✉️ Contact the Author

<ul>
<li>Author's Email: yonglelolu@foxmail.com</li>
<li>Author's WeChat: Downloader_Tools</li>
<li>WeChat Official Account: Downloader Tools</li>
<li><b>Discord Community</b>: <a href="https://discord.com/invite/ZYtmgKud9Y">Join Community</a></li>
<li>QQ Group Chat (Project Exchange): <a href="https://github.com/JoeanAmier/TikTokDownloader/blob/master/docs/QQ%E7%BE%A4%E8%81%8A%E4%BA%8C%E7%BB%B4%E7%A0%81.png">Scan to Join</a></li>
</ul>
<p>✨ <b>The author's other open-source projects:</b></p>
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