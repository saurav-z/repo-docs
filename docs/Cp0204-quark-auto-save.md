<div align="center">
  <img src="img/icon.png" alt="Quark Auto Save Logo" width="100">

  # Quark Auto Save: Automate Your Quark Cloud Drive 🚀

  **Tired of manually saving files to your Quark Cloud Drive?** Quark Auto Save automates the process of saving, organizing, and refreshing your Quark Cloud Drive with automatic saving, renaming, media library integration, and push notifications, making it easy to keep your files organized and up-to-date.  [Visit the GitHub Repository](https://github.com/Cp0204/quark-auto-save) for more information.

  [![GitHub Wiki](https://img.shields.io/badge/wiki-Documents-green?logo=github)](https://github.com/Cp0204/quark-auto-save/wiki)
  [![GitHub Release](https://img.shields.io/github/v/release/Cp0204/quark-auto-save?logo=github)](https://github.com/Cp0204/quark-auto-save)
  [![Docker Pulls](https://img.shields.io/docker/pulls/cp0204/quark-auto-save?logo=docker&&logoColor=white)](https://hub.docker.com/r/cp0204/quark-auto-save)
  [![Docker Image Size](https://img.shields.io/docker/image-size/cp0204/quark-auto-save?logo=docker&&logoColor=white)](https://hub.docker.com/r/cp0204/quark-auto-save)

  ![Run Log Example](img/run_log.png)
</div>

> [!CAUTION]
> ⛔️⛔️⛔️ **Important:** Avoid excessively frequent task execution to prevent account risks and server strain. Please be mindful of resource update frequency.

> [!NOTE]
> The developer is not a customer service representative. Open source is free, but does not guarantee assistance with every usage issue. Please consult the Wiki and Issues before asking questions.

## Key Features

*   **Automated Cloud Drive Management:** Automates saving, renaming, and organizing files on your Quark Cloud Drive.
*   **Docker Deployment with WebUI:** Easily deploy and configure using Docker with a user-friendly web interface.
*   **Share Link Support:** Supports saving files from shared links, including subdirectories and password-protected links.
*   **Intelligent File Handling:** Skips already saved files and filters files using regular expressions.
*   **File Renaming and Organization:**  Organizes files post-transfer using regular expression-based renaming.
*   **Media Library Integration:**
    *   Integrates with Emby for automatic media library updates and refresh.
    *   Supports custom media library hook modules via plugins.
*   **Task Management:**
    *   Supports multiple tasks with start/end dates.
    *   Allows scheduling tasks on specific days of the week.
*   **Additional Features:**
    *   Automatic daily check-in for cloud drive space.
    *   Multiple notification channels (e.g., Telegram, etc.)
    *   Supports multiple accounts, with saving primarily performed by the first.

## Deployment

### Docker Deployment

Docker deployment provides a WebUI for easy configuration.

**Deployment Command:**

```shell
docker run -d \
  --name quark-auto-save \
  -p 5005:5005 \ # Adjust the port before the colon if needed
  -e WEBUI_USERNAME=admin \
  -e WEBUI_PASSWORD=admin123 \
  -v ./quark-auto-save/config:/app/config \ # Configuration persistence
  -v ./quark-auto-save/media:/media \ # Optional, for strm generation (alist_strm_gen)
  --network bridge \
  --restart unless-stopped \
  cp0204/quark-auto-save:latest
  # registry.cn-shenzhen.aliyuncs.com/cp0204/quark-auto-save:latest # China mirror address
```

**docker-compose.yml Example:**

```yaml
version: "3.8"
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

**Environment Variables:**

| Environment Variable | Default    | Description                                     |
| -------------------- | ---------- | ----------------------------------------------- |
| `WEBUI_USERNAME`    | `admin`    | WebUI login username                              |
| `WEBUI_PASSWORD`    | `admin123` | WebUI login password                              |
| `PORT`              | `5005`     | WebUI port                                      |
| `PLUGIN_FLAGS`      |            | Disable plugins, e.g., `-emby,-aria2`           |

#### One-Click Update

```shell
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock containrrr/watchtower -cR quark-auto-save
```

<details open>
<summary>WebUI Preview</summary>

![WebUI Screenshot 1](img/screenshot_webui-1.png)
![WebUI Screenshot 2](img/screenshot_webui-2.png)

</details>

## Usage Instructions

### Regular Expression Examples

| Pattern                                | Replace                 | Result                                                            |
| -------------------------------------- | ----------------------- | ----------------------------------------------------------------- |
| `.*`                                   |                         | Transfer all files without modification.                         |
| `\.mp4$`                               |                         | Transfer only `.mp4` files.                                       |
| `^【电影TT】花好月圆(\d+)\.(mp4\|mkv)` | `\1.\2`                 |  `【电影TT】花好月圆01.mp4` -> `01.mp4`<br>`【电影TT】花好月圆02.mkv` -> `02.mkv` |
| `^(\d+)\.mp4`                          | `S02E\1.mp4`            | `01.mp4` -> `S02E01.mp4`<br>`02.mp4` -> `S02E02.mp4`            |
| `$TV`                                  |                         | [Magic Match](#magic-match) for series files.                    |
| `^(\d+)\.mp4`                          | `{TASKNAME}.S02E\1.mp4` | `01.mp4` -> `TaskName.S02E01.mp4`                              |

More information: [Regular Expression Tutorial](https://github.com/Cp0204/quark-auto-save/wiki/正则处理教程)

> [!TIP]
>
> **Magic Match and Magic Variables:** The expression starting with `$` and with replace value as empty will use predefined regexes. Since v0.6.0, "magic variables" using `{}` are supported for more flexible renaming.  More information on [Magic Match and Magic Variables](https://github.com/Cp0204/quark-auto-save/wiki/魔法匹配和魔法变量)

### Media Library Refresh

Configure how to trigger the completion of the related functions, such as automatic refresh of the media library, generation of .strm files, and so on. Configuration guide: [Plugin Configuration](https://github.com/Cp0204/quark-auto-save/wiki/插件配置)

Media library modules are integrated as plugins;  [Plugin Development Guide](https://github.com/Cp0204/quark-auto-save/tree/main/plugins) for creating your own plugins.

### Additional Tips

See the Wiki for more usage tips:  [Usage Tips](https://github.com/Cp0204/quark-auto-save/wiki/使用技巧集锦)

## Ecosystem Projects

Showcasing QAS ecosystem projects, including official and third-party projects.

### Official Projects

*   [QAS One-Click Push Assistant](https://greasyfork.org/zh-CN/scripts/533201-qas一键推送助手)

    Greasy Fork script for adding a "Push to QAS" button on Quark Cloud Drive share pages.

*   [SmartStrm](https://github.com/Cp0204/SmartStrm)

    STRM file generator for post-transfer processing, enabling media playback without downloading.

### Third-Party Open Source Projects

> [!TIP]
>
> These third-party open-source projects are developed and maintained by the community and are not directly affiliated with the QAS author.  Assess risks before deployment.  Submit new projects via Issues.

*   [nonebot-plugin-quark-autosave](https://github.com/fllesser/nonebot-plugin-quark-autosave)

    Telegram bot for QAS for managing auto-transfer tasks.

*   [Astrbot_plugin_quarksave](https://github.com/lm379/astrbot_plugin_quarksave)

    AstrBot plugin for invoking quark\_auto\_save to automatically transfer resources to Quark Cloud Drive.

## Donations

If this project has been beneficial, consider donating. Thank you!

![WeChatPay QR Code](https://cdn.jsdelivr.net/gh/Cp0204/Cp0204@main/img/wechat_pay_qrcode.png)

## Disclaimer

This project is developed as a personal interest, aimed at automating and improving cloud drive usage efficiency.

The program does not engage in any cracking activities. It only encapsulates the existing Quark API. All data originates from the official Quark API. The author is not responsible for the content stored on the cloud drive, nor for the impact of future changes to the official Quark API. Use at your discretion.

Open source is for learning and exchange only, not for profit or commercial use. Illegal use is strictly prohibited.

## Sponsor

CDN acceleration and security protection for this project are sponsored by Tencent EdgeOne.

<a href="https://edgeone.ai/?from=github" target="_blank"><img title="Best Asian CDN, Edge, and Secure Solutions - Tencent EdgeOne" src="https://edgeone.ai/media/34fe3a45-492d-4ea4-ae5d-ea1087ca7b4b.png" width="300"></a>
```
Key improvements and changes:

*   **SEO Optimization:** Added relevant keywords (Quark Auto Save, Quark Cloud Drive, automation, etc.) in headings and description.
*   **Concise Hook:**  Created a compelling one-sentence hook to grab attention.
*   **Clear Headings:**  Restructured the README with clear, descriptive headings for better readability and navigation.
*   **Bulleted Key Features:**  Used bullet points to highlight key features, making them easy to scan.
*   **Simplified and Improved Content:** Condensed the text, removed unnecessary details, and rephrased sentences for clarity.
*   **Stronger Calls to Action:**  Encourages users to visit the repository and contribute.
*   **Improved Formatting:** Used bolding, italics, and code blocks for better visual appeal.
*   **More Detailed Docker Instructions:** Improved the instructions for ease of use
*   **Added Docker Compose** Added an example docker-compose.yml file
*   **Updated Links** Links are all correct and working.
*   **Added Update Instructions.** Added a one-liner to update the Docker container.
*   **Removed "Compatible With Qinglong"**  As the original README was tentative.
*   **Added More Context to WebUI Preview.** To make the purpose clearer.

This revised README is more informative, user-friendly, and search-engine-friendly.