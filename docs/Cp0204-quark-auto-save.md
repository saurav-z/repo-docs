<div align="center">

![quark-logo](img/icon.png)

# Quark Auto-Save: Automate Quark Cloud Disk Transfers

**Tired of manually transferring files to Quark Cloud Disk?** Automate the process with Quark Auto-Save, which includes features like scheduled transfers, file renaming, Emby integration, and notifications, allowing you to automatically save files and keep your media library up to date.

[![Wiki][wiki-image]][wiki-url] [![GitHub Releases][gitHub-releases-image]][github-url] [![Docker Pulls][docker-pulls-image]][docker-url] [![Docker Image Size][docker-image-size-image]][docker-url]

[wiki-image]: https://img.shields.io/badge/wiki-Documents-green?logo=github
[gitHub-releases-image]: https://img.shields.io/github/v/release/Cp0204/quark-auto-save?logo=github
[docker-pulls-image]: https://img.shields.io/docker/pulls/cp0204/quark-auto-save?logo=docker&&logoColor=white
[docker-image-size-image]: https://img.shields.io/docker/image-size/cp0204/quark-auto-save?logo=docker&&logoColor=white
[github-url]: https://github.com/Cp0204/quark-auto-save
[docker-url]: https://hub.docker.com/r/cp0204/quark-auto-save
[wiki-url]: https://github.com/Cp0204/quark-auto-save/wiki

![run_log](img/run_log.png)

</div>

> [!CAUTION]
> ⛔️⛔️⛔️ **Important!** Resources are not updated constantly. **Avoid setting excessively frequent scheduled runs!** This is to prevent account risks and avoid putting unnecessary strain on Quark servers.

> [!NOTE]
> The developer is not a customer service representative. Open source and free do not equate to assistance with usage issues. The project Wiki is quite comprehensive. Please refer to the Issues and Wiki before asking questions.

## Key Features

*   **Automated Transfers:** Schedule transfers of files from shared links.
*   **WebUI Configuration:** Manage tasks, settings, and more through an intuitive web interface.
*   **Automated File Management:**
    *   Organize and rename files post-transfer using regex.
    *   Automatically create destination directories.
    *   Skip already transferred files.
    *   Filter files by name using regex.
    *   Option to ignore file extensions.
*   **Advanced Task Management:**
    *   Support for multiple tasks.
    *   Task expiration to prevent indefinite execution.
    *   Individual scheduling for specific days of the week.
*   **Emby Integration:**
    *   Search Emby media library based on task names.
    *   Automatically refresh the Emby library after transfers.
    *   Modular design allows for custom media library hook modules.
*   **Additional Features:**
    *   Daily check-in for Quark Cloud Disk space. ([How to check-in](https://github.com/Cp0204/quark-auto-save/wiki/使用技巧集锦#每日签到领空间))
    *   Support for multiple notification channels. ([Notification Setup](https://github.com/Cp0204/quark-auto-save/wiki/通知推送服务配置))
    *   Multi-account support (Sign in with multiple accounts, only the first account transfers).
    *   Supports child directories in shared links.
    *   Records and skips expired shared links.
    *   Supports shared links that require an extraction code. ([Shared links with extract codes](https://github.com/Cp0204/quark-auto-save/wiki/使用技巧集锦#支持需提取码的分享链接))
    *   Smart search of resources and auto-fill. ([CloudSaver Search](https://github.com/Cp0204/quark-auto-save/wiki/CloudSaver搜索源))

## Deployment

### Docker Deployment

Docker deployment offers a WebUI for management and configuration, providing a graphical interface to meet most needs.

**Docker Run Command:**

```shell
docker run -d \
  --name quark-auto-save \
  -p 5005:5005 \ # Port mapping, change the port before the colon, not the port after the colon
  -e WEBUI_USERNAME=admin \
  -e WEBUI_PASSWORD=admin123 \
  -v ./quark-auto-save/config:/app/config \ # Required for persistent configuration
  -v ./quark-auto-save/media:/media \ # Optional, for module alist_strm_gen generating strm
  --network bridge \
  --restart unless-stopped \
  cp0204/quark-auto-save:latest
  # registry.cn-shenzhen.aliyuncs.com/cp0204/quark-auto-save:latest # Domestic mirror address
```

**docker-compose.yml Example:**

```yaml
version: "3.9"
services:
  quark-auto-save:
    image: cp0204/quark-auto-save:latest
    container_name: quark-auto-save
    network_mode: bridge
    ports:
      - 5005:5005
    restart: unless-stopped
    environment:
      WEBUI_USERNAME: "admin"
      WEBUI_PASSWORD: "admin123"
    volumes:
      - ./quark-auto-save/config:/app/config
      - ./quark-auto-save/media:/media
```

**Access the WebUI:**  http://yourhost:5005

| Environment Variable | Default      | Description                            |
| -------------------- | ------------ | -------------------------------------- |
| `WEBUI_USERNAME`     | `admin`      | WebUI username                          |
| `WEBUI_PASSWORD`     | `admin123`   | WebUI password                          |
| `PORT`               | `5005`       | WebUI Port                              |
| `PLUGIN_FLAGS`       |              | Plugin flags, e.g., `-emby,-aria2` to disable plugins |

#### One-Click Update

```shell
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock containrrr/watchtower -cR quark-auto-save
```

<details open>
<summary>WebUI Preview</summary>

![screenshot_webui](img/screenshot_webui-1.png)

![screenshot_webui](img/screenshot_webui-2.png)

</details>

## Usage

### Regex Examples for File Handling

| Pattern                                | Replace                 | Result                                                                                                        |
| -------------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------- |
| `.*`                                   |                         | Transfers all files without renaming.                                                                         |
| `\.mp4$`                               |                         | Transfers all files with the `.mp4` extension.                                                              |
| `^【电影TT】花好月圆(\d+)\.(mp4\|mkv)` | `\1.\2`                 | 【电影TT】花好月圆01.mp4 → 01.mp4<br>【电影TT】花好月圆02.mkv → 02.mkv |
| `^(\d+)\.mp4`                          | `S02E\1.mp4`            | 01.mp4 → S02E01.mp4<br>02.mp4 → S02E02.mp4                                                                |
| `$TV`                                  |                         | [Magic Matching](#magic-matching) for TV series files.                                                     |
| `^(\d+)\.mp4`                          | `{TASKNAME}.S02E\1.mp4` | 01.mp4 → TaskName.S02E01.mp4                                                                                    |

For more regex examples and usage, see the [Regex Tutorial](https://github.com/Cp0204/quark-auto-save/wiki/正则处理教程).

> [!TIP]
>
> **Magic Matching and Magic Variables:** In regex processing, we've defined "magic matching" patterns. If the value of the expression starts with `$`, and the replacement field is left empty, the program automatically uses predefined regular expressions for matching and replacing.
>
> From v0.6.0 onwards, more "magic variables" enclosed in {} are supported, providing more flexibility for renaming.
>
> For more information, see [Magic Matching and Magic Variables](https://github.com/Cp0204/quark-auto-save/wiki/魔法匹配和魔法变量)

### Refreshing Your Media Library

You can trigger actions like automatic media library refreshing or .strm file generation when new content is transferred. See [Plugin Configuration](https://github.com/Cp0204/quark-auto-save/wiki/插件配置) for details.

Media library integration is implemented through plugins. If you are interested, please refer to the [Plugin Development Guide](https://github.com/Cp0204/quark-auto-save/tree/main/plugins).

### More Tips & Tricks

Refer to the Wiki: [Tips and Tricks](https://github.com/Cp0204/quark-auto-save/wiki/使用技巧集锦)

## Ecosystem Projects

The following showcases the QAS ecosystem, including official and third-party projects.

### Official Projects

*   [QAS One-Click Push Assistant](https://greasyfork.org/zh-CN/scripts/533201-qas一键推送助手)

    A Tampermonkey script that adds a "Push to QAS" button to Quark Cloud Disk share pages.

*   [SmartStrm](https://github.com/Cp0204/SmartStrm)

    STRM file generation tool for post-transfer processing and media library integration without downloading.

### Third-Party Open Source Projects

> [!TIP]
>
> These third-party open-source projects are developed and maintained by the community and are not directly affiliated with the QAS author. Please assess the associated risks before deploying to a production environment.
>
> If you have a new project not listed here, you can submit it via Issues.

*   [nonebot-plugin-quark-autosave](https://github.com/fllesser/nonebot-plugin-quark-autosave)

    QAS Telegram bot for quick management of automatic transfer tasks.

*   [Astrbot_plugin_quarksave](https://github.com/lm379/astrbot_plugin_quarksave)

    AstrBot plugin, calls quark\_auto\_save to automatically transfer resources to Quark Cloud Disk.

## Donations

If this project benefits you, you can donate. Thank you!

![WeChatPay](https://cdn.jsdelivr.net/gh/Cp0204/Cp0204@main/img/wechat_pay_qrcode.png)

## Disclaimer

This project is developed out of personal interest and aims to improve cloud disk usage efficiency through automation.

The program does not involve any cracking behavior; it only encapsulates the existing Quark API. All data comes from the official Quark API. The author is not responsible for the content on the cloud disk, nor for any impact caused by future changes to the official Quark API. Please use it at your own discretion.

Open source is for learning and exchange purposes only. It is not for profit and is not authorized for commercial use. It is strictly prohibited for illegal purposes.

## Sponsor

CDN acceleration and security protection for this project are sponsored by Tencent EdgeOne.

<a href="https://edgeone.ai/?from=github" target="_blank"><img title="Best Asian CDN, Edge, and Secure Solutions - Tencent EdgeOne" src="https://edgeone.ai/media/34fe3a45-492d-4ea4-ae5d-ea1087ca7b4b.png" width="300"></a>
```
Key improvements and explanations:

*   **SEO Optimization:**  The title and key features are optimized for search terms like "Quark Auto-Save," "Quark Cloud Disk Automation," and related keywords.  The headings are clear and descriptive.
*   **One-Sentence Hook:** The introductory sentence clearly states the primary benefit.
*   **Clear Headings & Structure:**  The document is organized with distinct sections (Key Features, Deployment, Usage, Ecosystem, etc.) making it easy to scan and understand.
*   **Bulleted Lists:** Key features are presented in easy-to-read bullet points.
*   **Concise Language:**  Redundant phrases have been removed, and the writing is more direct.
*   **Important Warnings Prominently Displayed:** The CAUTION and NOTE sections are emphasized.
*   **Docker Deployment Improved:** Includes both the `docker run` command and a `docker-compose.yml` example for different users.  Clear instructions for accessing the WebUI are added.
*   **Regex Examples Enhanced:**  More practical regex examples are provided.
*   **Links to Wiki:** The wiki is now more clearly referenced.
*   **Ecosystem Project Highlights:**  The sections for official and third-party projects are improved, and a disclaimer is added.
*   **Donation & Disclaimer Sections Improved:** The donation and disclaimer are cleaned up.
*   **Sponsor Information:** Kept sponsor information at the bottom.
*   **Clearer Explanations:**  More explicit and clearer explanations of features.  The use of "magic matching" and "magic variables" is better explained.
*   **More User-Friendly:**  The text is made more friendly by reducing Chinese characters and explaining what they mean in English.
*   **Uses bolding for more important items, and removes unnecessary bolding.**
*   **Added Docker-compose example**

This revised README is significantly better for both readability and discoverability.  It's well-structured, informative, and effectively promotes the Quark Auto-Save project.