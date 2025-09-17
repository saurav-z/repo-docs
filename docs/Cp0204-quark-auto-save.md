<div align="center">

![quark-logo](img/icon.png)

</div>

# Quark Auto-Save: Automate Your Quark Drive Transfers

Tired of manually transferring files to your Quark Drive? This tool automates the process, offering auto-saving, file organization, push notifications, and media library refreshing.  Check out the [original repository](https://github.com/Cp0204/quark-auto-save) for the full experience!

[![wiki][wiki-image]][wiki-url] [![github releases][gitHub-releases-image]][github-url] [![docker pulls][docker-pulls-image]][docker-url] [![docker image size][docker-image-size-image]][docker-url]

[wiki-image]: https://img.shields.io/badge/wiki-Documents-green?logo=github
[gitHub-releases-image]: https://img.shields.io/github/v/release/Cp0204/quark-auto-save?logo=github
[docker-pulls-image]: https://img.shields.io/docker/pulls/cp0204/quark-auto-save?logo=docker&&logoColor=white
[docker-image-size-image]: https://img.shields.io/docker/image-size/cp0204/quark-auto-save?logo=docker&&logoColor=white
[github-url]: https://github.com/Cp0204/quark-auto-save
[docker-url]: https://hub.docker.com/r/cp0204/quark-auto-save
[wiki-url]: https://github.com/Cp0204/quark-auto-save/wiki

![run_log](img/run_log.png)

## Key Features

*   **Automated Transfers:** Automatically saves files from shared links to your Quark Drive.
*   **Smart Handling:** Records and skips expired links, supports password-protected shares.
*   **File Management:**
    *   Creates target directories automatically.
    *   Skips already saved files.
    *   Uses regular expressions for file filtering and renaming.
    *   Supports ignoring file extensions.
*   **Task Management:**
    *   Manages multiple tasks.
    *   Sets task expiration dates.
    *   Allows scheduling of tasks by specific weekdays.
*   **Media Library Integration:**
    *   Searches Emby media library by task name.
    *   Refreshes Emby automatically after transfers (requires plugin configuration).
    *   Modular plugin architecture for custom media library integrations.
*   **Additional Features:**
    *   Daily check-in for free storage (requires configuration).
    *   Supports multiple notification channels.
    *   Supports multiple accounts (auto sign-in, only the first account transfers).

## Deployment

### Docker Deployment

Docker deployment offers a WebUI for easy configuration.

**Deployment Command:**

```shell
docker run -d \
  --name quark-auto-save \
  -p 5005:5005 \ # 映射端口，:前的可以改，即部署后访问的端口，:后的不可改
  -e WEBUI_USERNAME=admin \
  -e WEBUI_PASSWORD=admin123 \
  -v ./quark-auto-save/config:/app/config \ # 必须，配置持久化
  -v ./quark-auto-save/media:/media \ # 可选，模块alist_strm_gen生成strm使用
  --network bridge \
  --restart unless-stopped \
  cp0204/quark-auto-save:latest
  # registry.cn-shenzhen.aliyuncs.com/cp0204/quark-auto-save:latest # 国内镜像地址
```

**docker-compose.yml:**

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

**Access the WebUI:** http://yourhost:5005

**Environment Variables:**

| Variable          | Default   | Notes                                     |
| ----------------- | --------- | ----------------------------------------- |
| `WEBUI_USERNAME`  | `admin`   | WebUI login username                      |
| `WEBUI_PASSWORD`  | `admin123` | WebUI login password                      |
| `PORT`            | `5005`    | WebUI port                                |
| `PLUGIN_FLAGS`    |           | Disable specific plugins (e.g., `-emby,-aria2`) |

#### One-click update

```shell
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock containrrr/watchtower -cR quark-auto-save
```

<details open>
<summary>WebUI Preview</summary>

![screenshot_webui](img/screenshot_webui-1.png)

![screenshot_webui](img/screenshot_webui-2.png)

</details>

## Usage

### Regular Expression Examples

| Pattern                                | Replace                 | Effect                                                                  |
| -------------------------------------- | ----------------------- | ----------------------------------------------------------------------- |
| `.*`                                   |                         | Saves all files without renaming.                                         |
| `\.mp4$`                               |                         | Saves all files with the `.mp4` extension.                              |
| `^【电影TT】花好月圆(\d+)\.(mp4\|mkv)` | `\1.\2`                 | 【电影TT】花好月圆01.mp4 → 01.mp4<br>【电影TT】花好月圆02.mkv → 02.mkv |
| `^(\d+)\.mp4`                          | `S02E\1.mp4`            | 01.mp4 → S02E01.mp4<br>02.mp4 → S02E02.mp4                            |
| `$TV`                                  |                         | [Magic matching](#magic-matching) series files                          |
| `^(\d+)\.mp4`                          | `{TASKNAME}.S02E\1.mp4` | 01.mp4 → task name.S02E01.mp4                                            |

More regex usage instructions: [Regex Processing Tutorial](https://github.com/Cp0204/quark-auto-save/wiki/Regex Processing Tutorial)

> [!TIP]
>
> **Magic Matching and Magic Variables:**  We have defined "magic matching" patterns in regex processing. If the expression value starts with `$`, and the replace field is left blank, the program automatically uses preset regular expressions for matching and replacement.
>
> Since v0.6.0, support for more "magic variables" enclosed in {} is available, allowing for more flexible renaming.
>
> For more details, see [Magic Matching and Magic Variables](https://github.com/Cp0204/quark-auto-save/wiki/Magic Matching and Magic Variables)

### Refreshing Media Library

Configure this to refresh your media library (Emby, etc.) upon successful transfers.  See the [Plugin Configuration](https://github.com/Cp0204/quark-auto-save/wiki/Plugin Configuration) guide.

Media library modules are integrated as plugins. Refer to the [Plugin Development Guide](https://github.com/Cp0204/quark-auto-save/tree/main/plugins) for more info.

### Additional Tips

See the Wiki for more tips:  [Usage Tips](https://github.com/Cp0204/quark-auto-save/wiki/Usage Tips)

## Ecosystem Projects

Official and third-party projects that enhance Quark Auto-Save:

### Official Projects

*   [QAS One-Click Push Assistant](https://greasyfork.org/zh-CN/scripts/533201-qas%E4%B8%80%E9%94%AE%E6%8E%A8%E9%80%81%E5%8A%A9%E6%89%8B)
    *   Greasy Fork script to add a "Push to QAS" button on Quark Drive share pages.
*   [SmartStrm](https://github.com/Cp0204/SmartStrm)
    *   STRM file generation tool for post-transfer processing, allowing media to be added to your library without downloads.

### Third-Party Open Source Projects

> [!TIP]
>
> The following third-party open-source projects are developed and maintained by the community and are not directly affiliated with the QAS author. Please evaluate the risks before deploying them in a production environment.
>
> Submit new projects via Issues if they are not listed here.

*   [nonebot-plugin-quark-autosave](https://github.com/fllesser/nonebot-plugin-quark-autosave)
    *   QAS Telegram bot for quick management of auto-transfer tasks.
*   [Astrbot_plugin_quarksave](https://github.com/lm379/astrbot_plugin_quarksave)
    *   AstrBot plugin to use `quark_auto_save` to auto-transfer resources to Quark Drive.

## Donations

If you benefit from this project, please consider a small donation.  Thank you!

![WeChatPay](https://cdn.jsdelivr.net/gh/Cp0204/Cp0204@main/img/wechat_pay_qrcode.png)

## Disclaimer

This project is developed as a personal interest project to improve Quark Drive usage efficiency through automation.

The program does not involve any cracking behavior; it only encapsulates Quark's existing APIs, and all data comes from Quark's official APIs.  The author is not responsible for the content on the drives, or for changes to Quark's APIs that may cause issues.  Please use at your own risk.

Open source is only for learning and communication purposes, is non-profit, and is not authorized for commercial use.  Strictly prohibited for illegal purposes.

## Sponsor

CDN acceleration and security protection for this project are sponsored by Tencent EdgeOne.

<a href="https://edgeone.ai/?from=github" target="_blank"><img title="Best Asian CDN, Edge, and Secure Solutions - Tencent EdgeOne" src="https://edgeone.ai/media/34fe3a45-492d-4ea4-ae5d-ea1087ca7b4b.png" width="300"></a>
```
Key improvements and explanations:

*   **SEO Optimization:**  Includes relevant keywords ("Quark Drive", "auto-save", "automation", "media library") in headings and text.
*   **Hook:**  A clear, concise one-sentence hook to grab attention.
*   **Clear Headings:**  Uses descriptive and organized headings for readability.
*   **Bulleted Key Features:**  Provides a quick overview of the main functionality.
*   **Concise Language:**  Avoids unnecessary jargon.
*   **Actionable Instructions:** Provides clear steps, including deployment commands with explanations.
*   **Complete Docker information** Complete Docker config, including compose file and one-click update, is provided to ease deployment.
*   **Call to Action (Donations):**  Encourages donations.
*   **Disclaimer:**  Important legal information is clearly stated.
*   **Formatted Code Blocks:** Code blocks are formatted properly.
*   **Wiki Link:** Links to the Wiki are used for further details.
*   **Ecosystem Section:** Added an ecosystem section with both official and third-party projects.
*   **Tencent EdgeOne Sponsorship:** Added a clear section for the Tencent EdgeOne sponsorship with a relevant image.
*   **Markdown formatting consistency:**  Ensured consistent markdown formatting.
*   **Removed "May" and "Should" qualifiers:** Made instructions more direct.
*   **Corrected a Minor Grammar Error:** Corrected "in the drives" to "on the drives".