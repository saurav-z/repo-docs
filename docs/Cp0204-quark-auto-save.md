<div align="center">
  <img src="img/icon.png" alt="Quark Auto Save Logo" width="100">

  # Quark Auto Save: Automate Quark Cloud Drive Transfers & More!

  **Tired of manually transferring files to your Quark Cloud Drive?** Quark Auto Save automates Quark Cloud Drive tasks, including automatic transfer, renaming, media library refreshing, and notification pushes.  Keep your Quark cloud drive organized effortlessly!  See the [original repo](https://github.com/Cp0204/quark-auto-save) for more information.

  [![wiki][wiki-image]][wiki-url] [![github releases][gitHub-releases-image]][github-url] [![docker pulls][docker-pulls-image]][docker-url] [![docker image size][docker-image-size-image]][docker-url]

  [wiki-image]: https://img.shields.io/badge/wiki-Documents-green?logo=github
  [gitHub-releases-image]: https://img.shields.io/github/v/release/Cp0204/quark-auto-save?logo=github
  [docker-pulls-image]: https://img.shields.io/docker/pulls/cp0204/quark-auto-save?logo=docker&&logoColor=white
  [docker-image-size-image]: https://img.shields.io/docker/image-size/cp0204/quark-auto-save?logo=docker&&logoColor=white
  [github-url]: https://github.com/Cp0204/quark-auto-save
  [docker-url]: https://hub.docker.com/r/cp0204/quark-auto-save
  [wiki-url]: https://github.com/Cp0204/quark-auto-save/wiki

  <img src="img/run_log.png" alt="Run Log Screenshot">

</div>

> [!CAUTION]
> ⛔️⛔️⛔️  Do not set extremely frequent schedules to avoid account risks and put unnecessary load on the Quark servers.

> [!NOTE]
> The developer is not a customer service representative. Open source and free does not equate to solving all usage problems. Please consult the Wiki and existing Issues before asking questions.

## Key Features

*   **Automated Transfer:** Automatically saves files from shared links to your Quark Cloud Drive.
*   **Share Link Support:** Handles subdirectories within share links and skips expired links.
*   **Password-Protected Links:** Supports share links that require an extraction code <sup>[?](https://github.com/Cp0204/quark-auto-save/wiki/使用技巧集锦#支持需提取码的分享链接)</sup>
*   **Smart Resource Search:** Automatically fills in missing file names using the search sources.
*   **File Management:**
    *   Creates target directories if they don't exist.
    *   Skips files already transferred.
    *   Filters files by regular expressions.
    *   Renames files after transfer using regular expressions.
    *   Optionally ignores file extensions.
*   **Task Management:**
    *   Supports multiple tasks.
    *   Task expiration date to stop running a task.
    *   Schedule tasks to run on specific days of the week.
*   **Media Library Integration:**
    *   Searches Emby media library based on task names.
    *   Automatically refreshes the Emby media library after transfer or renaming.
    *   Modular media library integration, allowing users to easily develop their own hook modules.
*   **Additional Features:**
    *   Daily check-in for free space <sup>[?](https://github.com/Cp0204/quark-auto-save/wiki/使用技巧集锦#每日签到领空间)</sup>
    *   Supports multiple notification channels <sup>[?](https://github.com/Cp0204/quark-auto-save/wiki/通知推送服务配置)</sup>
    *   Supports multiple accounts (for daily sign-in; transfers only from the first account).

## Deployment

### Docker Deployment

Docker deployment provides a WebUI for configuration, with graphical configuration meeting most needs.

```shell
docker run -d \
  --name quark-auto-save \
  -p 5005:5005 \ # Port mapping, the port before : can be changed, i.e. the port accessed after deployment, the port after : cannot be changed
  -e WEBUI_USERNAME=admin \
  -e WEBUI_PASSWORD=admin123 \
  -v ./quark-auto-save/config:/app/config \ # Required, configuration persistence
  -v ./quark-auto-save/media:/media \ # Optional, module alist_strm_gen generates strm for use
  --network bridge \
  --restart unless-stopped \
  cp0204/quark-auto-save:latest
  # registry.cn-shenzhen.aliyuncs.com/cp0204/quark-auto-save:latest # Domestic mirror address
```

docker-compose.yml

```yaml
name: quark-auto-save
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

Manage WebUI at: `http://yourhost:5005`

| Environment Variable | Default      | Description                               |
| -------------------- | ------------ | ----------------------------------------- |
| `WEBUI_USERNAME`     | `admin`      | Administrator account                     |
| `WEBUI_PASSWORD`     | `admin123`   | Administrator password                    |
| `PORT`               | `5005`       | Management background port                |
| `PLUGIN_FLAGS`       |              | Plugin flags, e.g., `-emby,-aria2` disable some plugins |

#### One-Click Update

```shell
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock containrrr/watchtower -cR quark-auto-save
```

<details open>
<summary>WebUI Preview</summary>

![screenshot_webui](img/screenshot_webui-1.png)

![screenshot_webui](img/screenshot_webui-2.png)

</details>

## Usage Guide

### Regular Expression Examples

| Pattern                                | Replace                 | Effect                                                                   |
| -------------------------------------- | ----------------------- | ---------------------------------------------------------------------- |
| `.*`                                   |                         | Transfer all files without any renaming.                               |
| `\.mp4$`                               |                         | Transfer all files with the `.mp4` extension.                         |
| `^【电影TT】花好月圆(\d+)\.(mp4\|mkv)` | `\1.\2`                 | 【电影TT】花好月圆01.mp4 → 01.mp4<br>【电影TT】花好月圆02.mkv → 02.mkv |
| `^(\d+)\.mp4`                          | `S02E\1.mp4`            | 01.mp4 → S02E01.mp4<br>02.mp4 → S02E02.mp4                             |
| `$TV`                                  |                         | [Magic Matching](#magic-matching) episode files.                         |
| `^(\d+)\.mp4`                          | `{TASKNAME}.S02E\1.mp4` | 01.mp4 → Task name.S02E01.mp4                                             |

More regex instructions: [Regular Expression Tutorial](https://github.com/Cp0204/quark-auto-save/wiki/正则处理教程)

> [!TIP]
>
> **Magic Matching and Magic Variables**: In regular expression processing, we define some "magic matching" patterns. If the value of the expression begins with `$` and the replace field is left blank, the program will automatically use the preset regular expression for matching and replacing.
>
> From v0.6.0 onwards, we support more, which I call "magic variables" wrapped in `{}` to perform renaming more flexibly.
>
> For more instructions, please see [Magic Matching and Magic Variables](https://github.com/Cp0204/quark-auto-save/wiki/魔法匹配和魔法变量)

### Refreshing the Media Library

The media library can be refreshed upon new transfers. Configuration Guide: [Plugin Configuration](https://github.com/Cp0204/quark-auto-save/wiki/插件配置)

The media library module is integrated as a plugin. If you are interested, please refer to [Plugin Development Guide](https://github.com/Cp0204/quark-auto-save/tree/main/plugins).

### More Usage Tips

Please refer to the Wiki: [Tips and Tricks](https://github.com/Cp0204/quark-auto-save/wiki/使用技巧集锦)

## Ecosystem Projects

The following are QAS ecosystem projects, including official and third-party projects.

### Official Projects

*   [QAS One-Click Push Assistant](https://greasyfork.org/zh-CN/scripts/533201-qas一键推送助手)

    Tampermonkey script, adds a push to QAS button on the Quark Cloud Drive sharing page.

*   [SmartStrm](https://github.com/Cp0204/SmartStrm)

    STRM file generator, used for post-transfer processing, media files can be added to the library for playback without downloading.

### Third-Party Open Source Projects

> [!TIP]
>
> The following third-party open-source projects are developed and maintained by the community, and are not directly affiliated with the QAS author. Please assess the relevant risks yourself before deploying to a production environment.
>
> If you have a new project not listed here, you can submit it via Issues.

*   [nonebot-plugin-quark-autosave](https://github.com/fllesser/nonebot-plugin-quark-autosave)

    QAS Telegram bot for quick management of automatic transfer tasks.

*   [Astrbot_plugin_quarksave](https://github.com/lm379/astrbot_plugin_quarksave)

    AstrBot plugin, calling quark_auto_save to achieve the automatic transfer of resources to Quark Cloud Drive.

## Donations

If this project has been helpful to you, you can donate 1 yuan to me to let me know that open source is valuable. Thank you!

![WeChatPay](https://cdn.jsdelivr.net/gh/Cp0204/Cp0204@main/img/wechat_pay_qrcode.png)

## Disclaimer

This project is developed for personal interest and aims to improve cloud drive usage efficiency through automation.

The program does not involve any cracking behavior and only encapsulates the existing APIs of Quark. All data comes from the Quark official API. I am not responsible for the content of the cloud drive and am not responsible for any impact caused by future changes to the Quark official API. Please use it at your own discretion.

Open source is for learning and communication only. It is not for profit and is not authorized for commercial use. It is strictly forbidden for illegal purposes.

## Sponsor

CDN acceleration and security protection for this project are sponsored by Tencent EdgeOne.

<a href="https://edgeone.ai/?from=github" target="_blank"><img title="Best Asian CDN, Edge, and Secure Solutions - Tencent EdgeOne" src="https://edgeone.ai/media/34fe3a45-492d-4ea4-ae5d-ea1087ca7b4b.png" width="300"></a>
```
Key improvements and explanations:

*   **SEO Optimization:**  Added a concise and compelling one-sentence hook at the beginning.  Included relevant keywords like "Quark Cloud Drive," "automatic transfer," and "automation."
*   **Clear Structure:**  Improved heading levels for better readability and SEO (H1 for main title, H2 for sections).
*   **Bulleted Key Features:**  Uses bullet points for easy scanning and highlights the main functionalities.
*   **Concise Language:**  Revised wording throughout to be more direct and efficient.
*   **Docker Instructions:**  Kept the Docker instructions, but improved the layout.
*   **Emphasis on Benefits:**  Focuses on the user benefits of automation (saving time, organization).
*   **Wiki Links:**  Made sure all links to the wiki were present and correct.
*   **Consolidated Information:**  Grouped related information together (e.g., all the regex examples).
*   **Removed Redundancy:** Cut out repetitive phrases.
*   **Removed Unnecessary Formatting:** Removed unnecessary HTML formatting.  Uses Markdown.
*   **Clearer Warnings:**  Simplified the warnings and notes for clarity.
*   **Call to Action (Implicit):** The features and benefits implicitly encourage users to try the project.
*   **Improved the Sponsor Section:** Changed to "sponsor" and linked properly.
*   **Added Description of the Project:**  Added the purpose of the project with more details.
*   **More Detailed Instructions:**  More clear instructions on how to use the project.
*   **Better Disclaimer:** Better Disclaimer for the project.