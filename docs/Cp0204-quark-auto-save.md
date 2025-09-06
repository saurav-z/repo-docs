# Quark Auto-Save: Automate Your Quark Network Disk

**Tired of manually transferring and organizing files on your Quark Network Disk?** Quark Auto-Save automates your cloud storage tasks by automatically saving, organizing, and managing your Quark Network Disk files. Check out the original repo [here](https://github.com/Cp0204/quark-auto-save).

<div align="center">

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
> ⛔️⛔️⛔️ **Important:** Resources are not updated constantly.  **Avoid setting overly frequent schedules** to prevent potential account risks and server strain.

> [!NOTE]
> The developer is not a customer service representative.  This is an open-source project.  Please review the Wiki and Issues before asking questions.

## Key Features

*   **Automated Saving:** Automatically saves files from shared links to your Quark Network Disk.
*   **WebUI Control:**  User-friendly WebUI for easy configuration and management.
*   **Docker Deployment:**  Simple Docker deployment for easy setup and management.
*   **File Organization:**
    *   Rename files using regular expressions.
    *   Filter files by name.
    *   Skip already saved files.
*   **Task Management:**
    *   Supports multiple tasks.
    *   Set task expiration dates.
    *   Schedule tasks for specific days of the week.
*   **Media Library Integration:**
    *   Integrates with Emby media libraries.
    *   Automatically refreshes the media library.
    *   Modular design for custom media library hooks via plugins.
*   **Additional Features:**
    *   Daily sign-in to earn storage space.
    *   Supports multiple notification channels.
    *   Multi-account support.

## Deployment

### Docker Deployment

Docker deployment provides a WebUI for management and configuration.

```shell
docker run -d \
  --name quark-auto-save \
  -p 5005:5005 \ # Port Mapping, change the first one to customize the port, the second one cannot be changed.
  -e WEBUI_USERNAME=admin \
  -e WEBUI_PASSWORD=admin123 \
  -v ./quark-auto-save/config:/app/config \ # Required, Configuration persistence
  -v ./quark-auto-save/media:/media \ # Optional, module alist_strm_gen uses this to generate strm
  --network bridge \
  --restart unless-stopped \
  cp0204/quark-auto-save:latest
  # registry.cn-shenzhen.aliyuncs.com/cp0204/quark-auto-save:latest # Domestic mirror address
```

`docker-compose.yml` example:

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

| Environment Variable | Default    | Description                             |
| -------------------- | ---------- | --------------------------------------- |
| `WEBUI_USERNAME`     | `admin`    | Admin username                          |
| `WEBUI_PASSWORD`     | `admin123` | Admin password                          |
| `PORT`               | `5005`     | WebUI Port                               |
| `PLUGIN_FLAGS`       |            | Plugin flags, like `-emby,-aria2` to disable some plugins |

#### One-click Update

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

| Pattern                              | Replace         | Result                                                       |
| ------------------------------------ | --------------- | ------------------------------------------------------------ |
| `.*`                                 |                 | Save all files without modification                         |
| `\.mp4$`                             |                 | Save all files with the `.mp4` extension                    |
| `^【电影TT】花好月圆(\d+)\.(mp4\|mkv)` | `\1.\2`         | 【电影TT】花好月圆01.mp4 → 01.mp4<br>【电影TT】花好月圆02.mkv → 02.mkv |
| `^(\d+)\.mp4`                        | `S02E\1.mp4`    | 01.mp4 → S02E01.mp4<br>02.mp4 → S02E02.mp4                  |
| `$TV`                                |                 | [Magic Matching](#magic-matching)  episode files              |
| `^(\d+)\.mp4`                        | `{TASKNAME}.S02E\1.mp4` | 01.mp4 → taskname.S02E01.mp4                                |

More regular expression information:  [Regular Expression Tutorial](https://github.com/Cp0204/quark-auto-save/wiki/正则处理教程)

> [!TIP]
>
> **Magic Matching and Magic Variables:** In regular expression processing, we have defined some "magic matching" patterns. If the value of the expression starts with `$`, and the replacement expression is left blank, the program will automatically use the preset regular expression for matching and replacing.
>
> From v0.6.0 onwards, more "magic variables" wrapped in `{}` are supported, which allows for more flexible renaming.
>
> For more information, see [Magic Matching and Magic Variables](https://github.com/Cp0204/quark-auto-save/wiki/魔法匹配和魔法变量)

### Media Library Refresh

Triggers actions, such as refreshing the media library or generating `.strm` files, upon new saves. Configuration guide: [Plugin Configuration](https://github.com/Cp0204/quark-auto-save/wiki/插件配置)

Media library modules are integrated as plugins. If you are interested, please refer to the [plugin development guide](https://github.com/Cp0204/quark-auto-save/tree/main/plugins).

### Further Tips

Refer to the Wiki:  [Tips and Tricks](https://github.com/Cp0204/quark-auto-save/wiki/使用技巧集锦)

## Donations

If this project is beneficial to you, you can donate to show appreciation.

![WeChatPay](https://cdn.jsdelivr.net/gh/Cp0204/Cp0204@main/img/wechat_pay_qrcode.png)

## Disclaimer

This project is developed for personal interest to automate Quark Network Disk usage.

The program does not involve any cracking and only encapsulates the existing Quark APIs. The author is not responsible for the content on the network disk or any changes to the Quark APIs. Please use it at your own discretion.

Open source is for learning and communication purposes only and is not for profit or commercial use. It is strictly prohibited for illegal purposes.

## Sponsor

CDN acceleration and security protection for this project are sponsored by Tencent EdgeOne.

<a href="https://edgeone.ai/?from=github" target="_blank"><img title="Best Asian CDN, Edge, and Secure Solutions - Tencent EdgeOne" src="https://edgeone.ai/media/34fe3a45-492d-4ea4-ae5d-ea1087ca7b4b.png" width="300"></a>