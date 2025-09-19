<div align="center">
<img src="./static/images/DouK-Downloader.png" alt="DouK-Downloader" height="256" width="256"><br>
<h1>DouK-Downloader</h1>
</div>
<br>

## Effortlessly Download TikTok and Douyin Videos with DouK-Downloader!

DouK-Downloader is a versatile, open-source tool for downloading videos, images, and data from TikTok and Douyin (抖音). Access [the original repository here](https://github.com/JoeanAmier/TikTokDownloader).

---

**Key Features:**

*   ✅ **Comprehensive Downloading:** Download videos, images, and data from TikTok and Douyin, including posts, likes, collections, live streams, and more.
*   ✅ **Account and Collection Downloads:** Batch download videos from user accounts and collections.
*   ✅ **High-Quality Downloads:** Get videos in the best available quality, including original TikTok videos and Douyin videos without watermarks.
*   ✅ **Multiple Download Modes:** Supports various download methods including by link, account, and collection.
*   ✅ **Data Saving:** Supports CSV/XLSX/SQLite formats to preserve data.
*   ✅ **Proxy Support:** Configure proxy settings for enhanced data scraping.
*   ✅ **Flexible File Handling:** Includes features like file renaming and the ability to set file size limits.
*   ✅ **Web API:** Supports a web API for programmatic access to the download and data extraction features.
*   ✅ **Docker Support:** Easily deploy and run the downloader using Docker.
*   ✅ **And much more!** Explore a comprehensive feature list in the details below.

<details>
<summary><b>Feature List (click to expand)</b></summary>

*   ✅ Download Douyin videos/images without watermarks
*   ✅ Download Douyin live streams
*   ✅ Download high-quality video files
*   ✅ Download TikTok videos in original quality
*   ✅ Download TikTok videos/images without watermarks
*   ✅ Download videos from Douyin/TikTok accounts (posts/likes/collections)
*   ✅ Extract detailed Douyin/TikTok account data
*   ✅ Batch download from video links
*   ✅ Multi-account download support
*   ✅ Automatic skipping of already downloaded files
*   ✅ Data persistence (CSV/XLSX/SQLite)
*   ✅ Download of dynamic/static cover images
*   ✅ Get Douyin live stream URLs
*   ✅ Get TikTok live stream URLs
*   ✅ Use FFmpeg for live stream downloads
*   ✅ Web UI interactive interface (Future Development)
*   ✅ Extract Douyin video comment data
*   ✅ Download Douyin collection works
*   ✅ Download TikTok collection works
*   ✅ Record like and collection statistics
*   ✅ Filter works by publication time
*   ✅ Incremental download of account works
*   ✅ Proxy support for data collection
*   ✅ Remote access from LAN
*   ✅ Collect Douyin account details
*   ✅ Update work statistics
*   ✅ Support custom account/collection identifiers
*   ✅ Automatic updates of account nicknames/identifiers
*   ✅ Deployment to private/public servers
*   ✅ Collect Douyin search data
*   ✅ Collect Douyin trending data
*   ✅ Record downloaded video IDs
*   ✅ Read cookie from browser
*   ✅ Support Web API calls
*   ✅ Multi-threaded video download
*   ✅ File integrity handling
*   ✅ Custom rule filtering for videos
*   ✅ Archive files by folder
*   ✅ Custom file size limits
*   ✅ Support for resuming interrupted downloads
*   ✅ Clipboard link monitoring for downloads
</details>

---

## Quick Start

1.  **Download and Run:** Download the executable from [Releases](https://github.com/JoeanAmier/TikTokDownloader/releases/latest) or [Actions](https://github.com/JoeanAmier/TikTokDownloader/actions) and run the `main` file.
2.  **Configure Cookie (Important):**  Use the [Cookie Extraction Tutorial](https://github.com/JoeanAmier/TikTokDownloader/blob/master/docs/Cookie%E8%8E%B7%E5%8F%96%E6%95%99%E7%A8%8B.md) to get and configure your cookie.
3.  **Start Downloading:**
    *   Select `Terminal Interactive Mode` > `Batch download link works (general)` >  `Manually enter the work link to be collected`.
    *   Enter the video link to download (TikTok may require additional setup, see documentation).
4.  **For More Details:** Consult the detailed documentation in the [Project Documentation](https://github.com/JoeanAmier/TikTokDownloader/wiki/Documentation).

---

## Screenshots

**Terminal Interactive Mode**

*  ![Terminal Mode Screenshot 1](docs/screenshot/终端交互模式截图CN1.png)
*  ![Terminal Mode Screenshot 2](docs/screenshot/终端交互模式截图CN2.png)
*  ![Terminal Mode Screenshot 3](docs/screenshot/终端交互模式截图CN3.png)

---

## Web API Mode

*  ![WebAPI Mode Screenshot 1](docs/screenshot/WebAPI模式截图CN1.png)
*  ![WebAPI Mode Screenshot 2](docs/screenshot/WebAPI模式截图CN2.png)

**API Call Example**

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

## Docker

1.  **Get Image:**
    *   Build:  Use the `Dockerfile` to build the image.
    *   Pull:  Use `docker pull joeanamier/tiktok-downloader` or `docker pull ghcr.io/joeanamier/tiktok-downloader`.
2.  **Create Container:**  `docker run --name <container_name> -p 5555:5555 -v tiktok_downloader_volume:/app/Volume -it <image_name>`
3.  **Run Container:** `docker start -i <container_name/container_id>` or `docker restart -i <container_name/container_id>`.

---

##  Contribute

Your contributions are welcome! See the [Contribution Guidelines](#-contribution-guide) section.

---
##  Support the Project

If DouK-Downloader is helpful to you, consider giving it a **Star** ⭐.  Thank you for your support!

| WeChat (WeChat) | Alipay (Alipay) |
|---|---|
| <img src="./docs/微信赞助二维码.png" alt="微信赞助二维码" height="200" width="200"> | <img src="./docs/支付宝赞助二维码.png" alt="支付宝赞助二维码" height="200" width="200"> |

If you wish, you can consider providing funding to provide additional support for **DouK-Downloader**!

---

## Sponsors

*   **[DartNode](https://dartnode.com)**
[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")

*   **[ZMTO](https://www.zmto.com/)**
    <a href="https://www.zmto.com/"><img src="https://console.zmto.com/templates/2019/dist/images/logo_dark.svg" alt="ZMTO"></a>
    <p><a href="https://www.zmto.com/">ZMTO</a>：Cloud infrastructure provider providing efficient solutions with reliable cutting-edge technology and professional support, and providing enterprise-level VPS infrastructure for eligible open source projects, supporting the sustainable development and innovation of the open source ecosystem.</p>

*   **[TikHub](https://tikhub.io/)**
    <p><a href="https://tikhub.io/">TikHub</a>: A third-party API service provider, specializing in providing APIs for various platforms.</p>
    <p>By signing in daily, users can get a small amount of usage credit for free. You can use my <strong>recommendation link</strong>: <a href="https://user.tikhub.io/users/signup?referral_code=ZrdH8McC">https://user.tikhub.io/users/signup?referral_code=ZrdH8McC</a> or <strong>recommendation code</strong>: `ZrdH8McC` to register and recharge to get a $2 credit!</p>

---

## Contact

*   Email: yonglelolu@foxmail.com
*   WeChat: Downloader_Tools
*   WeChat Official Account: Downloader Tools
*   Discord Community: [Join the Community](https://discord.com/invite/ZYtmgKud9Y)
*   QQ Group (Project Discussion): [Scan to Join QQ Group](https://github.com/JoeanAmier/TikTokDownloader/blob/master/docs/QQ%E7%BE%A4%E8%81%8A%E4%BA%8C%E7%BB%B4%E7%A0%81.png)

---

## Other Open Source Projects by the Author

*   **XHS-Downloader (小红书、XiaoHongShu、RedNote):** [https://github.com/JoeanAmier/XHS-Downloader](https://github.com/JoeanAmier/XHS-Downloader)
*   **KS-Downloader (快手、KuaiShou):** [https://github.com/JoeanAmier/KS-Downloader](https://github.com/JoeanAmier/KS-Downloader)

---

## Star History
<p>
<img alt="Star History Chart" src="https://api.star-history.com/svg?repos=JoeanAmier/TikTokDownloader&amp;type=Timeline"/>
</p>

---

## Project References

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
---

## ⚠️ Disclaimer

*   The user is solely responsible for their use of this project and assumes all associated risks. The author is not liable for any losses, liabilities, or risks arising from the user's use of this project.
*   The code and features provided by the author are based on existing knowledge and technological developments. While the author strives to ensure the correctness and security of the code to the best of their ability, they do not guarantee that the code is entirely free from errors or defects.
*   All third-party libraries, plugins, or services used by this project are subject to their original open-source or commercial licenses, which users must review and comply with. The author is not responsible for the stability, security, or compliance of any third-party components.
*   Users must strictly adhere to the requirements of the <a href="https://github.com/JoeanAmier/TikTokDownloader/blob/master/LICENSE">GNU General Public License v3.0</a> when using this project, and must indicate that the code uses code from <a href="https://github.com/JoeanAmier/TikTokDownloader/blob/master/LICENSE">GNU General Public License v3.0</a> in the appropriate places.
*   Users must independently study relevant laws and regulations when using the code and features of this project, and ensure that their usage is legal and compliant. The user is solely responsible for any legal liabilities and risks arising from violations of laws and regulations.
*   Users must not use this tool for any activities that infringe on intellectual property rights, including but not limited to unauthorized downloading and distribution of copyrighted content. The developer does not participate in, support, or endorse the acquisition or distribution of any illegal content.
*   This project is not responsible for the compliance of data collection, storage, transmission, and other processing activities by users. Users should comply with relevant laws and regulations and ensure that their processing activities are legal and just; legal liabilities arising from violations of regulations shall be borne by the user.
*   Users shall not, under any circumstances, associate the author, contributors, or other related parties of this project with the user's usage behavior, or require them to be responsible for any losses or damages resulting from the user's use of this project.
*   The author of this project will not provide paid versions of the DouK-Downloader project, nor will they provide any commercial services related to the DouK-Downloader project.
*   Any secondary development, modification, or compilation of this project by users is not related to the original author, and the original author is not responsible for any responsibility related to the secondary development behavior or its results; users should be fully responsible for all kinds of situations that may be caused by secondary development.
*   This project does not grant users any patent licenses; if the use of this project leads to patent disputes or infringement, users shall bear all risks and responsibilities on their own. Without the written authorization of the author or the rights holder, it is not allowed to use this project for any commercial promotion, promotion or re-authorization.
*   The author reserves the right to terminate the service to any user who violates this statement at any time, and may require them to destroy the obtained code and derivative works.
*   The author reserves the right to update this statement without prior notice, and the continued use of the user implies acceptance of the revised terms.

<b>Before using the code and features of this project, please carefully consider and accept the above disclaimer. If you have any questions or disagree with the above statement, please do not use the code and features of this project. If you use the code and features of this project, it is deemed that you have fully understood and accepted the above disclaimer, and voluntarily assume all risks and consequences of using this project.</b>

---

## 🌟 Contribution Guide

**We welcome contributions to this project! Please read the following guidelines carefully to ensure your contributions are successfully accepted and integrated, in order to maintain a clean, efficient, and easily maintainable codebase.**

*   Before starting development, please pull the latest code from the `develop` branch and use it as the basis for modification; this helps to avoid merge conflicts and ensures that your changes are based on the latest project state.
*   If your changes involve multiple unrelated features or issues, please separate them into multiple independent commits or pull requests.
*   Each pull request should focus on a single feature or fix as much as possible, for the convenience of code review and testing.
*   Follow the existing code style; please ensure that your code is consistent with the existing code style in the project; it is recommended to use the Ruff tool to maintain the code format specification.
*   Write readable code; add appropriate comments to help others understand your intentions.
*   Each commit should contain a clear and concise commit message to describe the changes made. Commit messages should follow this format: `<Type>: <Brief description>`
*   When you are ready to submit a pull request, please give priority to submitting them to the `develop` branch; this is to give the maintainer a buffer to perform additional testing and review before the final merge into the `master` branch.
*   It is recommended to communicate with the author before development or when you have any questions to ensure that the development direction is consistent, and avoid redundant work or invalid submissions.

**Reference:**

*   [Contributor Covenant](https://www.contributor-covenant.org/zh-cn/version/2/1/code_of_conduct/)
*   [How to Contribute to Open Source](https://opensource.guide/zh-hans/how-to-contribute/)