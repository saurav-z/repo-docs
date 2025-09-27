<div align="center">

![quark-logo](img/icon.png)

# Quark Auto-Save: Automate Your Quark Drive and Never Miss a File!

Automate your Quark Drive with Quark Auto-Save, saving you time and effort by automatically transferring files, organizing them, and refreshing your media library.  Check out the [original repo](https://github.com/Cp0204/quark-auto-save) for more details.

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
> ⛔️⛔️⛔️ **Important:** Avoid excessively frequent scheduled runs to prevent account risks and avoid putting unnecessary strain on Quark's servers.

> [!NOTE]
> The developer is not a customer service representative. This is an open-source, free project, and does not guarantee solutions for all usage problems. Please consult the Issues and Wiki first before asking questions.

## Key Features

*   **Automated File Transfers:** Automatically transfer files from shared links to your Quark Drive.
*   **Share Link Support:** Handles subdirectories within shared links, identifies and skips invalid links, and supports links requiring passwords.
*   **Smart File Management:** Automatically creates target directories, skips already saved files, filters files by name (using regular expressions), and organizes filenames after transfer.
*   **Flexible Task Management:** Supports multiple tasks, task expiration dates, and the ability to schedule tasks for specific days of the week.
*   **Media Library Integration:** Search your Emby media library based on task names and automatically refresh your Emby media library after file transfers.
*   **Additional Features:** Includes daily check-in for storage space, multiple notification channels, and support for multiple accounts.

## Deployment

### Docker Deployment

Docker provides a WebUI for easy configuration.  Here's the basic deployment command:

```shell
docker run -d \
  --name quark-auto-save \
  -p 5005:5005 \ # Port mapping, change the part before the colon (:) to your desired port
  -e WEBUI_USERNAME=admin \
  -e WEBUI_PASSWORD=admin123 \
  -v ./quark-auto-save/config:/app/config \ # Required for configuration persistence
  -v ./quark-auto-save/media:/media \ # Optional, for generating .strm files via alist_strm_gen
  --network bridge \
  --restart unless-stopped \
  cp0204/quark-auto-save:latest
  # registry.cn-shenzhen.aliyuncs.com/cp0204/quark-auto-save:latest # China Mirror
```

docker-compose.yml example:

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

Access the web UI:  http://yourhost:5005

| Environment Variable | Default     | Description                                    |
| -------------------- | ----------- | ---------------------------------------------- |
| `WEBUI_USERNAME`     | `admin`     | WebUI Username                                 |
| `WEBUI_PASSWORD`     | `admin123`  | WebUI Password                                 |
| `PORT`               | `5005`      | WebUI Port                                     |
| `PLUGIN_FLAGS`       |             | Plugin flags, e.g., `-emby,-aria2` to disable plugins |

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

| Pattern                                  | Replace                 | Effect                                                                                                    |
| ---------------------------------------- | ----------------------- | --------------------------------------------------------------------------------------------------------- |
| `.*`                                     |                         | Transfers all files without any changes.                                                                   |
| `\.mp4$`                                 |                         | Transfers all files with the `.mp4` extension.                                                           |
| `^【电影TT】花好月圆(\d+)\.(mp4\|mkv)` | `\1.\2`                 | 【电影TT】花好月圆01.mp4 → 01.mp4<br>【电影TT】花好月圆02.mkv → 02.mkv                                      |
| `^(\d+)\.mp4`                            | `S02E\1.mp4`            | 01.mp4 → S02E01.mp4<br>02.mp4 → S02E02.mp4                                                              |
| `$TV`                                    |                         | [Magic Match](#magic-match) for TV series files.                                                          |
| `^(\d+)\.mp4`                            | `{TASKNAME}.S02E\1.mp4` | 01.mp4 → TaskName.S02E01.mp4                                                                              |

For more on regular expression usage: [Regular Expression Tutorial](https://github.com/Cp0204/quark-auto-save/wiki/%E6%AD%A3%E5%88%99%E5%A4%84%E7%90%86%E6%95%99%E7%A8%8B)

> [!TIP]
>
> **Magic Matching and Magic Variables:** In regular expression processing, we define "magic matching" patterns. If the *Expression* value begins with `$` and the *Replacement* is left blank, the program will automatically use preset regular expressions for matching and replacing.
>
> From v0.6.0 onwards, support for more "{}" enclosed variables (Magic Variables), allowing for more flexible renaming.
>
> See [Magic Matching and Magic Variables](https://github.com/Cp0204/quark-auto-save/wiki/%E9%AD%94%E6%B3%95%E5%8C%B9%E9%85%8D%E5%92%8C%E9%AD%94%E6%B3%95%E5%8F%98%E9%87%8F) for details.

### Refresh Media Library

The media library can be refreshed after transfers by setting up plugins.  Refer to the plugin configuration guide: [Plugin Configuration](https://github.com/Cp0204/quark-auto-save/wiki/%E6%8F%92%E4%BB%B6%E9%85%8D%E7%BD%AE)

Media library modules are integrated as plugins.  See the [Plugin Development Guide](https://github.com/Cp0204/quark-auto-save/tree/main/plugins) if you wish to create your own.

### Additional Tips

See the Wiki for more tips: [Usage Tips](https://github.com/Cp0204/quark-auto-save/wiki/%E4%BD%BF%E7%94%A8%E6%8A%80%E5%B7%A7%E9%9B%86%E9%94%A6)

## Ecosystem Projects

These are projects that work alongside QAS.

### Official Projects

*   [QAS One-Click Push Assistant](https://greasyfork.org/zh-CN/scripts/533201-qas%E4%B8%80%E9%94%AE%E6%8E%A8%E9%80%81%E5%8A%A9%E6%89%8B)

    A Greasemonkey script that adds a button to Quark Drive share pages to push files to QAS.

*   [SmartStrm](https://github.com/Cp0204/SmartStrm)

    STRM file generation tool for post-transfer processing, enabling direct playback without downloading.

### Third-Party Open Source Projects

> [!TIP]
>
> The following third-party open-source projects are developed and maintained by the community and are not directly affiliated with the QAS author. Please assess any associated risks before deploying them in a production environment.
>
> If you have a project that is not listed, you can submit it via Issues.

*   [nonebot-plugin-quark-autosave](https://github.com/fllesser/nonebot-plugin-quark-autosave)

    QAS Telegram bot for managing auto-transfer tasks.

*   [Astrbot_plugin_quarksave](https://github.com/lm379/astrbot_plugin_quarksave)

    AstrBot plugin that uses quark\_auto\_save to automatically transfer resources to Quark Drive.

## Donate

If you find this project beneficial, consider donating to support its development.

![WeChatPay](https://cdn.jsdelivr.net/gh/Cp0204/Cp0204@main/img/wechat_pay_qrcode.png)

## Disclaimer

This project is developed as a personal interest, aimed at automating Quark Drive usage.

The program does not involve any cracking behavior, but only encapsulates Quark's existing APIs. All data comes from Quark's official APIs. The author is not responsible for the content on the drive or any impact caused by potential changes to the Quark API. Please use at your own discretion.

This open-source project is for learning and communication only, not for profit or commercial use. Illegal use is strictly prohibited.

## Sponsor

CDN acceleration and security protection for this project are sponsored by Tencent EdgeOne.

<a href="https://edgeone.ai/?from=github" target="_blank"><img title="Best Asian CDN, Edge, and Secure Solutions - Tencent EdgeOne" src="https://edgeone.ai/media/34fe3a45-492d-4ea4-ae5d-ea1087ca7b4b.png" width="300"></a>