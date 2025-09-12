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

## 🚀 Effortlessly Download TikTok and Douyin Content with DouK-Downloader

DouK-Downloader is your all-in-one solution for downloading TikTok and Douyin videos, images, and more.  Visit the [original repository](https://github.com/JoeanAmier/TikTokDownloader) for more details.

---

## ✨ Key Features

*   ✅ **Download TikTok & Douyin Content:** Videos, images, collections, and more!
*   ✅ **Account Scraping:** Download content from TikTok and Douyin accounts (posts, likes, favorites).
*   ✅ **High-Quality Downloads:** Get videos in the best available resolution.
*   ✅ **Batch Downloads:** Download multiple videos at once.
*   ✅ **API Support:** Utilize the Web API for integration.
*   ✅ **User-Friendly Interface:** Includes both terminal and Web UI (Web UI is currently under development).
*   ✅ **Docker Support:** Deploy and run the downloader within a Docker container.
*   ✅ **Proxy Support:** Use proxies for data acquisition.
*   ✅ **Cross-Platform Compatibility:** Works on Windows, macOS, and Linux.

---

## 💻 Project Screenshots

### Terminal Mode

![Terminal Mode Screenshot 1](docs/screenshot/终端交互模式截图CN1.png)
<br/>
![Terminal Mode Screenshot 2](docs/screenshot/终端交互模式截图CN2.png)
<br/>
![Terminal Mode Screenshot 3](docs/screenshot/终端交互模式截图CN3.png)

### Web API Mode

> **Access the API Documentation at: `http://127.0.0.1:5555/docs` or `http://127.0.0.1:5555/redoc`**

![Web API Mode Screenshot 1](docs/screenshot/WebAPI模式截图CN1.png)
<br/>
![Web API Mode Screenshot 2](docs/screenshot/WebAPI模式截图CN2.png)

#### API Example

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

## 📚 Quick Start

1.  **Download:** Download the pre-built executable from the [Releases](https://github.com/JoeanAmier/TikTokDownloader/releases/latest) or [Actions](https://github.com/JoeanAmier/TikTokDownloader/actions) page.
2.  **Run:** Run the executable (e.g., `main` on macOS/Windows).
3.  **Configure Cookie:**  Configure the Cookie to access the API from the instructions provided in the [Cookie Getting Tutorial](https://github.com/JoeanAmier/TikTokDownloader/blob/master/docs/Cookie%E8%8E%B7%E5%8F%96%E6%95%99%E7%A8%8B.md).
4.  **Start Downloading:** In Terminal Mode, select the appropriate options, then enter the video link.
5.  **For detailed instructions and further customization:**  See the full documentation on the [Project Wiki](https://github.com/JoeanAmier/TikTokDownloader/wiki/Documentation).

### 🐳 Docker

1.  **Get Image:**  Build your image by using the `Dockerfile` file or pulling the image from `docker pull joeanamier/tiktok-downloader` or `docker pull ghcr.io/joeanamier/tiktok-downloader`.
2.  **Create Container:** `docker run --name <container_name> -p <host_port>:5555 -v tiktok_downloader_volume:/app/Volume -it <image_name>`
3.  **Run Container:**
    *   `docker start -i <container_name/container_ID>`
    *   `docker restart -i <container_name/container_ID>`

    **Note:** Some features are unavailable in Docker due to limitations.

---

## 🔐 Cookie Information

*   [Cookie Getting Tutorial](https://github.com/JoeanAmier/TikTokDownloader/blob/master/docs/Cookie%E8%8E%B7%E5%8F%96%E6%95%99%E7%A8%8B.md)
*   Update your cookie if the video resolution is poor.
*   Update your cookie if the data acquisition fails.

---

## ⚠️ Disclaimer & Contribution Guidelines

**(The original Disclaimer, Contribution Guidelines, and Support Information have been retained.  These are found in the original README and were not altered.)**