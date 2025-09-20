<div align="center">
  <img src="img/icon.png" alt="Quark Auto Save Logo" width="100">
</div>

# Quark Auto Save: Automate Your Quark Network Disk Transfers

**Automatically transfer, organize, and manage your Quark Network Disk files with ease.** This tool simplifies the process of saving files from Quark Network Disk, ensuring your content is always up-to-date.

[![GitHub Releases](https://img.shields.io/github/v/release/Cp0204/quark-auto-save?logo=github)](https://github.com/Cp0204/quark-auto-save)
[![Docker Pulls](https://img.shields.io/docker/pulls/cp0204/quark-auto-save?logo=docker&&logoColor=white)](https://hub.docker.com/r/cp0204/quark-auto-save)
[![Docker Image Size](https://img.shields.io/docker/image-size/cp0204/quark-auto-save?logo=docker&&logoColor=white)](https://hub.docker.com/r/cp0204/quark-auto-save)
[![Wiki](https://img.shields.io/badge/wiki-Documents-green?logo=github)](https://github.com/Cp0204/quark-auto-save/wiki)

<img src="img/run_log.png" alt="Run Log Screenshot">

> [!CAUTION]
> **Important:** Do not set excessively high execution frequencies to avoid potential account risks and server strain.

> [!NOTE]
> The developer is not a customer service representative. Open source is free, but does not guarantee solutions to all usage problems. Please consult the Wiki and Issues before asking questions.

## Key Features

*   **Automated Transfers:** Automatically saves files from Quark Network Disk.
*   **WebUI Management:** Easy-to-use web interface for configuration and control.
*   **Docker Deployment:** Simple and convenient Docker setup for easy use.
*   **Share Link Support:** Handles subdirectories and supports share links requiring extraction codes.
*   **File Management:** Renames and organizes files after transfer, filters files, and skips already transferred files.
*   **Media Library Integration:** Integrates with Emby and can automatically refresh the media library after new transfers.
*   **Task Scheduling:** Supports multiple tasks with individual scheduling options.
*   **Notification Support:** Integrates with multiple notification channels.
*   **Account Support:** Supports multiple accounts.

## Installation and Usage

### Docker Deployment

Docker deployment provides a WebUI management and graphical configuration. To deploy, run:

```shell
docker run -d \
  --name quark-auto-save \
  -p 5005:5005 \ # Port mapping, change the port before : as needed
  -e WEBUI_USERNAME=admin \
  -e WEBUI_PASSWORD=admin123 \
  -v ./quark-auto-save/config:/app/config \ # Required, config persistence
  -v ./quark-auto-save/media:/media \ # Optional, for alist_strm_gen
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

Access the web UI at: `http://yourhost:5005`

| Environment Variable | Default     | Notes                                       |
| -------------------- | ----------- | ------------------------------------------- |
| `WEBUI_USERNAME`     | `admin`     | Admin account                               |
| `WEBUI_PASSWORD`     | `admin123`  | Admin password                              |
| `PORT`               | `5005`      | Admin backend port                          |
| `PLUGIN_FLAGS`       |             | Plugin flags, such as `-emby,-aria2` to disable plugins |

#### One-Click Update

```shell
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock containrrr/watchtower -cR quark-auto-save
```

<details open>
<summary>WebUI Preview</summary>

![screenshot_webui](img/screenshot_webui-1.png)

![screenshot_webui](img/screenshot_webui-2.png)

</details>

## Configuration and Customization

### Regular Expression Examples

| Pattern                                | Replace              | Result                                                            |
| -------------------------------------- | -------------------- | ----------------------------------------------------------------- |
| `.*`                                   |                      | Transfers all files without modification.                         |
| `\.mp4$`                               |                      | Transfers all files with the `.mp4` extension.                    |
| `^【电影TT】花好月圆(\d+)\.(mp4\|mkv)` | `\1.\2`              | 【电影TT】花好月圆01.mp4 → 01.mp4<br>【电影TT】花好月圆02.mkv → 02.mkv |
| `^(\d+)\.mp4`                          | `S02E\1.mp4`         | 01.mp4 → S02E01.mp4<br>02.mp4 → S02E02.mp4                         |
| `$TV`                                  |                      | [Magic Matching](#magic-matching) for TV series files              |
| `^(\d+)\.mp4`                          | `{TASKNAME}.S02E\1.mp4` | 01.mp4 → TaskName.S02E01.mp4                                      |

More Regex instructions: [Regex Processing Tutorial](https://github.com/Cp0204/quark-auto-save/wiki/%E6%AD%A3%E5%88%99%E5%A4%84%E7%90%86%E6%95%99%E7%A8%8B)

> [!TIP]
>
> **Magic Matching and Magic Variables:** We have defined "magic matching" patterns in regex processing. If the value of the expression starts with `$ ` and the replace expression is left blank, the program will automatically use the preset regular expression for matching and replacing.
>
> From v0.6.0, support for more "magic variables" enclosed in {} is supported, which can be used for more flexible renaming.
>
> For more information, see [Magic Matching and Magic Variables](https://github.com/Cp0204/quark-auto-save/wiki/%E9%AD%94%E6%B3%95%E5%8C%B9%E9%85%8D%E5%92%8C%E9%AD%94%E6%B3%95%E5%8F%98%E9%87%8F)

### Refresh Media Library

Refresh your media library automatically by configuring the plugin, such as Emby or Plex. Configuration guide: [Plugin Configuration](https://github.com/Cp0204/quark-auto-save/wiki/%E6%8F%92%E4%BB%B6%E9%85%8D%E7%BD%AE)

The media library module is integrated as a plugin. If you are interested, please refer to the [Plugin Development Guide](https://github.com/Cp0204/quark-auto-save/tree/main/plugins).

### More Tips

For more tips, please refer to the Wiki: [Tips and Tricks](https://github.com/Cp0204/quark-auto-save/wiki/%E4%BD%BF%E7%94%A8%E6%8A%80%E5%B7%A7%E9%9B%86%E9%94%A6)

## Ecosystem Projects

The following shows QAS ecosystem projects, including official and third-party projects.

### Official Projects

*   [QAS One-Click Push Assistant](https://greasyfork.org/zh-CN/scripts/533201-qas%E4%B8%80%E9%94%AE%E6%8E%A8%E9%80%81%E5%8A%A9%E6%89%8B)

    Tampermonkey script, adds a button to push to QAS on the Quark Network Disk sharing page.

*   [SmartStrm](https://github.com/Cp0204/SmartStrm)

    STRM file generation tool, for post-processing after transfer, media without download library playback.

### Third-Party Open Source Projects

> [!TIP]
>
> The following third-party open source projects are developed and maintained by the community and are not directly affiliated with the QAS author. Before deploying to a production environment, please evaluate the relevant risks yourself.
>
> If you have a new project that is not listed here, you can submit it through Issues.

*   [nonebot-plugin-quark-autosave](https://github.com/fllesser/nonebot-plugin-quark-autosave)

    QAS Telegram bot, quickly manages auto-transfer tasks

*   [Astrbot_plugin_quarksave](https://github.com/lm379/astrbot_plugin_quarksave)

    AstrBot plugin, calls quark_auto_save to achieve automatic transfer of resources to the Quark Network Disk

## Donate

If this project has benefited you, you can donate a small amount to show your appreciation. Thank you!

![WeChatPay](https://cdn.jsdelivr.net/gh/Cp0204/Cp0204@main/img/wechat_pay_qrcode.png)

## Disclaimer

This project is developed out of personal interest and aims to improve the efficiency of using network disks through automation.

The program does not involve any cracking behavior and is merely an encapsulation of existing Quark APIs. The author is not responsible for the content on the network disk, nor for the impact of any future changes to the Quark APIs. Please use it at your own discretion.

Open source is for learning and communication purposes only, it is not for profit and is not authorized for commercial use. It is strictly forbidden to use it for illegal purposes.

## Sponsor

CDN acceleration and security protection for this project are sponsored by Tencent EdgeOne.

<a href="https://edgeone.ai/?from=github" target="_blank"><img title="Best Asian CDN, Edge, and Secure Solutions - Tencent EdgeOne" src="https://edgeone.ai/media/34fe3a45-492d-4ea4-ae5d-ea1087ca7b4b.png" width="300"></a>

For more information, see the original repository: [https://github.com/Cp0204/quark-auto-save](https://github.com/Cp0204/quark-auto-save)