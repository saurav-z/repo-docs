# Quark Auto-Save: Automate Your Quark Cloud Drive

**Tired of manually saving files to Quark cloud drive? Quark Auto-Save automates the process of saving, organizing, and refreshing your media library, making content management a breeze.** [View the original repository](https://github.com/Cp0204/quark-auto-save).

[![Wiki](https://img.shields.io/badge/wiki-Documents-green?logo=github)](https://github.com/Cp0204/quark-auto-save/wiki)
[![GitHub Releases](https://img.shields.io/github/v/release/Cp0204/quark-auto-save?logo=github)](https://github.com/Cp0204/quark-auto-save)
[![Docker Pulls](https://img.shields.io/docker/pulls/cp0204/quark-auto-save?logo=docker&&logoColor=white)](https://hub.docker.com/r/cp0204/quark-auto-save)
[![Docker Image Size](https://img.shields.io/docker/image-size/cp0204/quark-auto-save?logo=docker&&logoColor=white)](https://hub.docker.com/r/cp0204/quark-auto-save)

![Run Log Example](img/run_log.png)

> [!CAUTION]
> **Important!** Avoid overly frequent scheduled runs to prevent account risks and server strain. Every action has consequences.

> [!NOTE]
> The developer is not customer service. This is open-source software, and support is not guaranteed. Refer to the Wiki and Issues before asking questions.

## Key Features

*   **Automated Saving:** Automatically transfers files from shared links to your Quark cloud drive.
*   **File Management:** Organizes files with features like renaming, filtering, and skipping already saved files.
*   **Emby Integration:** Integrates seamlessly with Emby media servers for automatic media library refreshes.
*   **Flexible Deployment:** Supports Docker deployment with a user-friendly WebUI and customizable configuration.
*   **Task Management:** Manages multiple tasks with individual scheduling options.
*   **Notifications:** Sends push notifications through various channels.
*   **Multiple Account Support:** Allows for multiple account sign-in and saving.
*   **Optional Daily Check-In:** Automated daily check-in for space.

## Deployment

### Docker Deployment

Docker provides WebUI for easy configuration.

```shell
docker run -d \
  --name quark-auto-save \
  -p 5005:5005 \ # Port mapping, change the first number, the second must be 5005
  -e WEBUI_USERNAME=admin \
  -e WEBUI_PASSWORD=admin123 \
  -v ./quark-auto-save/config:/app/config \ # Required, config persistence
  -v ./quark-auto-save/media:/media \ # Optional, used by module alist_strm_gen
  --network bridge \
  --restart unless-stopped \
  cp0204/quark-auto-save:latest
  # registry.cn-shenzhen.aliyuncs.com/cp0204/quark-auto-save:latest # China mirror
```

`docker-compose.yml`

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

Access the management interface: `http://yourhost:5005`

| Environment Variable | Default      | Notes                                  |
| -------------------- | ------------ | -------------------------------------- |
| `WEBUI_USERNAME`     | `admin`      | Admin account                          |
| `WEBUI_PASSWORD`     | `admin123`   | Admin password                         |
| `PORT`               | `5005`       | Management interface port              |
| `PLUGIN_FLAGS`       |              | Plugins flags, e.g. `-emby,-aria2` disable plugins |

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

| Pattern                                | Replace                 | Effect                                                                   |
| -------------------------------------- | ----------------------- | ---------------------------------------------------------------------- |
| `.*`                                   |                         | Transfer all files, no renaming                                               |
| `\.mp4$`                               |                         | Transfer all files with the `.mp4` extension                                             |
| `^【电影TT】花好月圆(\d+)\.(mp4\|mkv)` | `\1.\2`                 | 【电影TT】花好月圆01.mp4 → 01.mp4<br>【电影TT】花好月圆02.mkv → 02.mkv |
| `^(\d+)\.mp4`                          | `S02E\1.mp4`            | 01.mp4 → S02E01.mp4<br>02.mp4 → S02E02.mp4                             |
| `$TV`                                  |                         | [Magic Matching](#magic-matching) episode files                                          |
| `^(\d+)\.mp4`                          | `{TASKNAME}.S02E\1.mp4` | 01.mp4 → TaskName.S02E01.mp4                                             |

Learn more: [Regular Expression Tutorial](https://github.com/Cp0204/quark-auto-save/wiki/正则处理教程)

> [!TIP]
>
> **Magic Matching and Variables:** The program will automatically match and replace using preset regular expressions if the `pattern` starts with `$` and `replace` is left blank.

> [!TIP]
> From v0.6.0 onwards, more "magic variables" enclosed in {} are supported for more flexible renaming.
>
> See [Magic Matching and Variables](https://github.com/Cp0204/quark-auto-save/wiki/魔法匹配和魔法变量) for more details.

### Media Library Refresh

Automatically triggers functions such as refreshing media libraries or generating `.strm` files when new transfers occur. Configuration guide: [Plugin Configuration](https://github.com/Cp0204/quark-auto-save/wiki/插件配置)

Media library integration is done through plugins. If you are interested, please refer to the [Plugin Development Guide](https://github.com/Cp0204/quark-auto-save/tree/main/plugins).

### Further Tips

Refer to the Wiki for more tips: [Tips and Tricks](https://github.com/Cp0204/quark-auto-save/wiki/使用技巧集锦)

## Donations

If you found this project helpful, consider a small donation. Thank you!

![WeChatPay](https://cdn.jsdelivr.net/gh/Cp0204/Cp0204@main/img/wechat_pay_qrcode.png)

## Disclaimer

This project is developed for personal interest, aimed at improving cloud disk efficiency through automation.

The program does not involve any cracking behavior and merely encapsulates existing Quark API, with all data sourced from the official Quark API. The developer is not responsible for the content on your cloud disk or potential changes to the Quark API. Please use the program at your own discretion.

Open source is for learning and communication purposes only, and is not for profit or commercial use, and is strictly prohibited from being used for illegal purposes.

## Sponsor

CDN acceleration and security protection for this project are sponsored by Tencent EdgeOne.

<a href="https://edgeone.ai/?from=github" target="_blank"><img title="Best Asian CDN, Edge, and Secure Solutions - Tencent EdgeOne" src="https://edgeone.ai/media/34fe3a45-492d-4ea4-ae5d-ea1087ca7b4b.png" width="300"></a>