<div align="center">
<img src="./static/images/DouK-Downloader.png" alt="DouK-Downloader" height="256" width="256"><br>
</div>

# DouK-Downloader: Download and Archive TikTok & Douyin Content with Ease

This powerful tool ([GitHub Repo](https://github.com/JoeanAmier/TikTokDownloader)) allows you to effortlessly download and archive content from TikTok and Douyin.

<div align="center">
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

## Key Features

*   **Comprehensive Content Downloading:** Download videos, images, and live streams from TikTok and Douyin, including:
    *   User posts (published, liked, favorites)
    *   Collections/Albums
    *   Live streams (download and archiving)
    *   Comments data
    *   Hotlist data
*   **High-Quality Downloads:** Obtain original and high-resolution video files.
*   **Account Data Scraping:** Extract detailed account information from TikTok and Douyin.
*   **Flexible Download Options:**
    *   Batch downloads of posts from accounts and collections.
    *   Download by direct link.
    *   Supports multiple accounts.
*   **Advanced Functionality:**
    *   Web UI and API support (in development).
    *   Automatic file organization.
    *   Proxy support for data collection.
    *   Cookie support (browser and clipboard).
    *   Docker Support

## Getting Started

1.  **Download:** Download the latest release from the [Releases](https://github.com/JoeanAmier/TikTokDownloader/releases/latest) or [Actions](https://github.com/JoeanAmier/TikTokDownloader/actions) page.
2.  **Configuration:** Configure your environment and install necessary dependencies
3.  **Cookie Setup:** Follow the instructions to configure the cookie. You can read more on [Cookie Setup](https://github.com/JoeanAmier/TikTokDownloader/blob/master/docs/Cookie%E8%8E%B7%E5%8F%96%E6%95%99%E7%A8%8B.md)
4.  **Start Downloading:** Use the command-line interface or Web UI to download your desired content.

For detailed instructions, refer to the [Project Documentation](https://github.com/JoeanAmier/TikTokDownloader/wiki/Documentation).

## Example API Usage

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

## Docker Usage
1.  **Get Images:** You can build it by using the Dockerfile, or pulling it with the command `docker pull joeanamier/tiktok-downloader`
2.  **Create Container:** `docker run --name <container_name> -p <host_port>:5555 -v tiktok_downloader_volume:/app/Volume -it <image_name>`
3.  **Run Container:** `docker start -i <container_name/container_id>`

## Support the Project

If DouK-Downloader is valuable to you, please consider giving it a ⭐ on GitHub! Your support is greatly appreciated.

## Sponsors

*   **DartNode**: <a href="https://dartnode.com "><img src="https://dartnode.com/branding/DN-Open-Source-sm.png" alt="DartNode" ></a>
*   **ZMTO**: <a href="https://www.zmto.com/"><img src="https://console.zmto.com/templates/2019/dist/images/logo_dark.svg" alt="ZMTO"></a>
*   **TikHub**: <a href="https://tikhub.io/">TikHub</a>

## Contact the Author

*   Email: yonglelolu@foxmail.com
*   WeChat: Downloader\_Tools
*   WeChat Public Account: Downloader Tools
*   Discord Community: [Join Discord](https://discord.com/invite/ZYtmgKud9Y)
*   QQ Group: [Scan QR Code](https://github.com/JoeanAmier/TikTokDownloader/blob/master/docs/QQ%E7%BE%A4%E8%81%8A%E4%BA%8C%E7%BB%B4%E7%A0%81.png)

## Other Projects

*   **XHS-Downloader (小红书)**: [XHS-Downloader Repo](https://github.com/JoeanAmier/XHS-Downloader)
*   **KS-Downloader (快手)**: [KS-Downloader Repo](https://github.com/JoeanAmier/KS-Downloader)

## Star History

<p>
<img alt="Star History Chart" src="https://api.star-history.com/svg?repos=JoeanAmier/TikTokDownloader&amp;type=Timeline"/>
</p>

## Project References

*   [F2](https://github.com/Johnserf-Seed/f2)
*   [TikTokDownload](https://github.com/Johnserf-Seed/TikTokDownload)
*   [Douyin_TikTok_Download_API](https://github.com/Evil0ctal/Douyin_TikTok_Download_API)
*   [TiktokDouyinCrawler](https://github.com/NearHuiwen/TiktokDouyinCrawler)
*   [DouyinLiveRecorder](https://github.com/ihmily/DouyinLiveRecorder)
*   [httpx](https://github.com/encode/httpx/)
*   [rich](https://github.com/Textualize/rich)
*   [aiosqlite](https://github.com/omnilib/aiosqlite)
*   [aiofiles](https://github.com/Tinche/aiofiles)
*   [rookie](https://github.com/thewh1teagle/rookie)
*   [pyinstaller](https://github.com/pyinstaller/pyinstaller)
*   [openpyxl](https://foss.heptapod.net/openpyxl/openpyxl)
*   [emoji](https://github.com/carpedm20/emoji/)
*   [lxml](https://github.com/lxml/lxml)
*   [ffmpeg](https://ffmpeg.org/ffmpeg-all.html)
*   [tikwm](https://www.tikwm.com/)

---