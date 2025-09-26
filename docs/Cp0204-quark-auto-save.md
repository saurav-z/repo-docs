<div align="center">

![quark-logo](img/icon.png)

# Quark Auto-Save: Automate Quark Drive Transfers and Media Library Integration

Tired of manually transferring and organizing files on Quark Drive? **Quark Auto-Save automates your Quark Drive workflow with features like auto-transfer, renaming, and media library refreshing.**

[![wiki][wiki-image]][wiki-url] [![github releases][gitHub-releases-image]][github-url] [![docker pulls][docker-pulls-image]][docker-url] [![docker image size][docker-image-size-image]][docker-url]

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
> ⛔️⛔️⛔️ **Important**: Resources don't update every second. Avoid setting overly frequent scheduling to prevent account risks and excessive strain on Quark servers. Every action has consequences!

> [!NOTE]
> The developer is not customer support. Open source is not synonymous with assistance in solving usage issues; please refer to the Wiki and Issues before asking for help.

## Key Features

*   **Automated Transfers:** Automatically transfers files from shared links to your Quark Drive.
*   **WebUI Configuration:** Easy-to-use WebUI for configuration and management.
*   **Smart File Handling:**
    *   Handles subdirectories in share links.
    *   Skips already transferred files.
    *   Filters files by regular expressions.
    *   Renames files after transfer using regular expressions.
    *   Option to ignore file extensions.
*   **Advanced Task Management:**
    *   Supports multiple tasks.
    *   Set task end dates.
    *   Schedule tasks to run on specific days.
*   **Media Library Integration:**
    *   Searches Emby media libraries by task name.
    *   Refreshes Emby media libraries after transfer.
    *   Modular design with plugin support for custom media library integrations.
*   **Additional Features:**
    *   Daily check-in for free storage.
    *   Supports multiple notification channels.
    *   Supports multiple accounts (transfers only from the primary account).

For further features, consult the [original repo](https://github.com/Cp0204/quark-auto-save).

## Deployment

### Docker Deployment

Docker deployment provides a WebUI for management and configuration.

```shell
docker run -d \
  --name quark-auto-save \
  -p 5005:5005 \ # Port mapping
  -e WEBUI_USERNAME=admin \
  -e WEBUI_PASSWORD=admin123 \
  -v ./quark-auto-save/config:/app/config \ # Configuration persistence
  -v ./quark-auto-save/media:/media \ # Optional, for alist_strm_gen module
  --network bridge \
  --restart unless-stopped \
  cp0204/quark-auto-save:latest
  # registry.cn-shenzhen.aliyuncs.com/cp0204/quark-auto-save:latest # China Mirror
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

Access the management interface at: `http://yourhost:5005`

| Environment Variable | Default       | Notes                                     |
| -------------------- | ------------- | ----------------------------------------- |
| `WEBUI_USERNAME`     | `admin`       | Admin account                             |
| `WEBUI_PASSWORD`     | `admin123`    | Admin password                            |
| `PORT`               | `5005`        | Management backend port                   |
| `PLUGIN_FLAGS`       |               | Plugin flags, such as `-emby,-aria2`     |

#### One-Click Update

```shell
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock containrrr/watchtower -cR quark-auto-save
```

<details open>
<summary>WebUI Preview</summary>

![screenshot_webui](img/screenshot_webui-1.png)

![screenshot_webui](img/screenshot_webui-2.png)

</details>

## Usage Instructions

### Regular Expression Examples

| Pattern                                | Replace                 | Effect                                                                                                   |
| -------------------------------------- | ----------------------- | -------------------------------------------------------------------------------------------------------- |
| `.*`                                   |                         | Transfer all files without modification.                                                               |
| `\.mp4$`                               |                         | Transfer all files with the `.mp4` extension.                                                           |
| `^【电影TT】花好月圆(\d+)\.(mp4\|mkv)` | `\1.\2`                 | 【电影TT】花好月圆01.mp4 → 01.mp4<br>【电影TT】花好月圆02.mkv → 02.mkv                                  |
| `^(\d+)\.mp4`                          | `S02E\1.mp4`            | 01.mp4 → S02E01.mp4<br>02.mp4 → S02E02.mp4                                                          |
| `$TV`                                  |                         | [Magic Matching](#magic-matching) Series file.                                                           |
| `^(\d+)\.mp4`                          | `{TASKNAME}.S02E\1.mp4` | 01.mp4 → Task name.S02E01.mp4                                                                           |

More regex instructions: [Regular Expression Tutorial](https://github.com/Cp0204/quark-auto-save/wiki/%E6%AD%A3%E5%88%99%E5%A4%84%E7%90%86%E6%95%99%E7%A8%8B)

> [!TIP]
>
> **Magic Matching and Magic Variables:** In regular expression processing, we've defined some "magic matching" patterns. If the value of the expression starts with $, and the replace expression is left blank, the program will automatically use predefined regular expressions for matching and replacement.
>
> Starting from v0.6.0, it supports more "magic variables" enclosed in {}, allowing for more flexible renaming.
>
> For more information, see [Magic Matching and Magic Variables](https://github.com/Cp0204/quark-auto-save/wiki/%E9%AD%94%E6%B3%95%E5%8C%B9%E9%85%8D%E5%92%8C%E9%AD%94%E6%B3%95%E5%8F%98%E9%87%8F)

### Refresh Media Library

After new transfers, trigger actions like automatically refreshing your media library or generating .strm files. See the configuration guide: [Plugin Configuration](https://github.com/Cp0204/quark-auto-save/wiki/%E6%8F%92%E4%BB%B6%E9%85%8D%E7%BD%AE)

Media library modules are integrated as plugins. If you're interested, refer to the [Plugin Development Guide](https://github.com/Cp0204/quark-auto-save/tree/main/plugins).

### More Tips

Please refer to the Wiki: [Usage Tips](https://github.com/Cp0204/quark-auto-save/wiki/%E4%BD%BF%E7%94%A8%E6%8A%80%E5%B7%A7%E9%9B%86%E9%94%A6)

## Ecosystem Projects

Below are QAS ecosystem projects, including official and third-party projects.

### Official Projects

*   [QAS One-Click Push Assistant](https://greasyfork.org/zh-CN/scripts/533201-qas%E4%B8%80%E9%94%AE%E6%8E%A8%E9%80%81%E5%8A%A9%E6%89%8B)

    Greasy Fork Script, adds a button to the Quark Drive sharing page to push to QAS.

*   [SmartStrm](https://github.com/Cp0204/SmartStrm)

    STRM file generation tool, for post-transfer processing, and media without downloading to the library for playback.

### Third-Party Open Source Projects

> [!TIP]
>
> The following third-party open-source projects are developed and maintained by the community and are not directly affiliated with the QAS author. Please assess any associated risks before deploying to a production environment.
>
> If you have a new project that isn't listed here, you can submit it via Issues.

*   [nonebot-plugin-quark-autosave](https://github.com/fllesser/nonebot-plugin-quark-autosave)

    QAS Telegram bot for quick management of automatic transfer tasks.

*   [Astrbot\_plugin\_quarksave](https://github.com/lm379/astrbot_plugin_quarksave)

    AstrBot plugin to call quark\_auto\_save for automatically transferring resources to Quark Drive.

## Donate

If this project has benefited you, you can donate a small amount. Thank you!

![WeChatPay](https://cdn.jsdelivr.net/gh/Cp0204/Cp0204@main/img/wechat_pay_qrcode.png)

## Disclaimer

This project is developed out of personal interest, aiming to improve the efficiency of using cloud drives through program automation.

The program does not involve any cracking behavior, only encapsulates the existing Quark API. All data comes from the official Quark API; I am not responsible for the content on the cloud drive or for the impact of potential future changes to the Quark API, please use it at your own discretion.

Open source is for learning and communication purposes only, is not for profit, and is not authorized for commercial use; it is strictly prohibited for illegal purposes.

## Sponsor

CDN acceleration and security protection for this project are sponsored by Tencent EdgeOne.

<a href="https://edgeone.ai/?from=github" target="_blank"><img title="Best Asian CDN, Edge, and Secure Solutions - Tencent EdgeOne" src="https://edgeone.ai/media/34fe3a45-492d-4ea4-ae5d-ea1087ca7b4b.png" width="300"></a>
```
Key changes and explanations:

*   **SEO-Optimized Title:** The main title includes "Quark Auto-Save" and a strong keyword phrase "Automate Quark Drive Transfers and Media Library Integration" to attract searches.
*   **One-Sentence Hook:** The one-sentence hook immediately clarifies the main benefit and purpose of the project.
*   **Clear Headings:**  Uses `##` and `###` for well-structured content organization.
*   **Bulleted Key Features:**  Provides a concise overview of capabilities for quick scanning.
*   **Links to Wiki and Repo:** Links are preserved, and the prominent mention of the original repo with "For further features, consult the [original repo](https://github.com/Cp0204/quark-auto-save)." increases direct clicks.
*   **Concise Deployment Instructions:** Docker instructions are presented concisely, but include vital information.  The Docker Compose example is extremely helpful.  One-click update instructions are included.
*   **Regular Expression Examples:** Gives clear examples of RegEx usage to aid with the initial user experience.
*   **Ecosystem Section:**  Highlights and credits other projects, which is valuable to the community.
*   **Disclaimer and Sponsor Sections:** Preserved for transparency and context.
*   **Markdown formatting:**  Uses bold, italics, and other Markdown features to improve readability.
*   **Removed "May" and "Should" wording** The instructions given in the code are definitive.
*   **Refined wording** Refined sentences in the original text for readability.
*   **Corrected typos** Corrected a few typos.