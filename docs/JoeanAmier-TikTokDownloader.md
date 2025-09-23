<div align="center">
<img src="./static/images/DouK-Downloader.png" alt="DouK-Downloader" height="256" width="256"><br>
<h1>DouK-Downloader: Download TikTok, Douyin, and More!</h1>
<p>Easily download videos, images, and data from TikTok, Douyin, and other platforms with this open-source tool.</p>
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

## Key Features:

*   ✅ Download TikTok and Douyin videos, images, and more.
*   ✅ Download videos from accounts, likes, and collections.
*   ✅ Download high-quality video files.
*   ✅ Supports batch downloading and multiple accounts.
*   ✅ Includes a Web UI and API for easy access.
*   ✅ Download comments and trending data.
*   ✅ Built-in proxy support and file integrity checks.
*   ✅ Support for Docker.
*   ✅ Customizable file saving and filtering options.

<hr>

## Table of Contents
*   [**Key Features**](#key-features)
*   [**Project Overview**](#project-overview)
*   [**Installation and Usage**](#installation-and-usage)
    *   [Quick Start](#quick-start)
    *   [Docker](#docker)
*   [**Getting Started**](#getting-started)
    *   [Cookie Information](#cookie-information)
    *   [Other Notes](#other-notes)
*   [**Contribute**](#contribute)
*   [**Support the Project**](#support-the-project)
*   [**Sponsors**](#sponsors)
*   [**Contact**](#contact)
*   [**Related Projects**](#related-projects)
*   [**Star History**](#star-history)
*   [**Project References**](#project-references)

<hr>

## Project Overview

DouK-Downloader is a powerful, open-source tool built to download videos, images, and data from popular platforms like TikTok and Douyin. It leverages the HTTPX module to provide a robust and free solution for data collection and file downloading.  This project, formerly known as `TikTokDownloader`, offers a wide array of features, including batch downloads, live stream recording, and data scraping.

<hr>

## Installation and Usage

### Quick Start

To get started with DouK-Downloader:

1.  **Download:**  Get the pre-compiled executable from the [Releases](https://github.com/JoeanAmier/TikTokDownloader/releases/latest) or [Actions](https://github.com/JoeanAmier/TikTokDownloader/actions) pages.  Alternatively, you can run the project by configuring the environment.
2.  **Run the executable:**  Unzip and run the `main` executable.
3.  **Configure Cookie:**  Configure the Cookie in the configuration file.
4.  **Download:** Enter the download link of the video that you want to download.

For a more detailed guide, including information on environment setup and Docker, see the [project documentation](https://github.com/JoeanAmier/TikTokDownloader/wiki/Documentation).

### Docker

1.  **Get the Image:** Use one of the following methods:
    *   Build from `Dockerfile`.
    *   `docker pull joeanamier/tiktok-downloader`
    *   `docker pull ghcr.io/joeanamier/tiktok-downloader`
2.  **Create Container:**  `docker run --name your_container_name -p host_port:5555 -v tiktok_downloader_volume:/app/Volume -it <image_name>`
3.  **Run Container:**
    *   Start: `docker start -i your_container_name/container_ID`
    *   Restart: `docker restart -i your_container_name/container_ID`

**Note:** Some features (like browser cookie reading) are limited in Docker.

<hr>

## Getting Started

### Cookie Information

*   Cookies are essential for accessing some features.
*   Update your cookie when it expires.
*   Higher resolution videos may require a current cookie.
*   Refer to the [Cookie Tutorial](https://github.com/JoeanAmier/TikTokDownloader/blob/master/docs/Cookie%E8%8E%B7%E5%8F%96%E6%95%99%E7%A8%8B.md) for instructions.

### Other Notes

*   Press Enter to go back in the menu, and 'Q' or 'q' to quit.
*   Account data download may take longer for accounts with lots of uploads.
*   Proxy configuration is set up in `settings.json`.
*   Use a JSON formatting tool for editing the configuration file.
*   Windows users need to run the program as an administrator to access browser cookies.

<hr>

## Contribute

We welcome contributions! Please read the [Contribution Guide](https://github.com/JoeanAmier/TikTokDownloader/blob/master/CONTRIBUTING.md) before getting started.

<hr>

## Support the Project

If you find `DouK-Downloader` helpful, please consider giving it a star ⭐ on GitHub!

| WeChat | Alipay |
| ----- | ----- |
| <img src="./docs/微信赞助二维码.png" alt="WeChat QR Code" height="200" width="200"> | <img src="./docs/支付宝赞助二维码.png" alt="Alipay QR Code" height="200" width="200"> |

You can also consider providing financial support.

<hr>

## Sponsors

### DartNode

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")

<hr>

### ZMTO

<a href="https://www.zmto.com/"><img src="https://console.zmto.com/templates/2019/dist/images/logo_dark.svg" alt="ZMTO"></a>
<p><a href="https://www.zmto.com/">ZMTO</a> provides a high-performance cloud infrastructure provider for open source projects.</p>

<hr>

### TikHub

<p><a href="https://tikhub.io/">TikHub</a> offers third-party API services.</p>
<p>Get free usage credits through daily check-ins. Register and recharge using my [referral link](https://user.tikhub.io/users/signup?referral_code=ZrdH8McC) or code: `ZrdH8McC` for a $2 credit!</p>

<hr>

## Contact

*   Email: yonglelolu@foxmail.com
*   WeChat: Downloader_Tools
*   WeChat Official Account: Downloader Tools
*   Discord Community:  [Join our Discord](https://discord.com/invite/ZYtmgKud9Y)
*   QQ Group:  <a href="https://github.com/JoeanAmier/TikTokDownloader/blob/master/docs/QQ%E7%BE%A4%E8%81%8A%E4%BA%8C%E7%BB%B4%E7%A0%81.png">Scan QR to join</a>

## Related Projects

*   **XHS-Downloader (小红书, XiaoHongShu, RedNote)**: [https://github.com/JoeanAmier/XHS-Downloader](https://github.com/JoeanAmier/XHS-Downloader)
*   **KS-Downloader (快手, KuaiShou)**: [https://github.com/JoeanAmier/KS-Downloader](https://github.com/JoeanAmier/KS-Downloader)

<hr>

## Star History

```html
<p>
<img alt="Star History Chart" src="https://api.star-history.com/svg?repos=JoeanAmier/TikTokDownloader&amp;type=Timeline"/>
</p>
```

<hr>

## Project References

*   [List of referenced projects from the original README]