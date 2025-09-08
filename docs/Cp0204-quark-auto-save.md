<div align="center">

![quark-logo](img/icon.png)

</div>

# Quark Auto-Save: Automate Your Quark Drive

**Tired of manually saving files to your Quark Drive?**  Quark Auto-Save is your all-in-one solution for automating Quark Drive tasks like auto-saving, renaming, and refreshing your media library.  You can find the original repo [here](https://github.com/Cp0204/quark-auto-save).

[![GitHub Release](https://img.shields.io/github/v/release/Cp0204/quark-auto-save?logo=github)](https://github.com/Cp0204/quark-auto-save)
[![Docker Pulls](https://img.shields.io/docker/pulls/cp0204/quark-auto-save?logo=docker&&logoColor=white)](https://hub.docker.com/r/cp0204/quark-auto-save)
[![Docker Image Size](https://img.shields.io/docker/image-size/cp0204/quark-auto-save?logo=docker&&logoColor=white)](https://hub.docker.com/r/cp0204/quark-auto-save)
[![Wiki](https://img.shields.io/badge/wiki-Documents-green?logo=github)](https://github.com/Cp0204/quark-auto-save/wiki)

![run_log](img/run_log.png)

> [!CAUTION]
> ⛔️⛔️⛔️ **Important:** Avoid excessive scheduling to prevent account risks and server strain.

> [!NOTE]
> The developer is not customer service. Refer to the Wiki and Issues for solutions before asking for help.

## Key Features

*   **Automated Saving:**  Automatically transfers files from share links to your Quark Drive.
*   **WebUI Configuration:** Easy setup and management via a user-friendly web interface (Docker).
*   **Smart File Handling:**
    *   Skips already saved files.
    *   Supports regex filtering for file selection.
    *   Renames files after saving using customizable regex rules.
    *   Option to ignore file extensions.
*   **Flexible Task Management:**
    *   Supports multiple tasks.
    *   Set task expiration dates.
    *   Schedule tasks for specific days of the week.
*   **Media Library Integration:**
    *   Searches and refreshes Emby media libraries based on task names.
    *   Modular plugin system for custom media library integration.
*   **Additional Features:**
    *   Daily check-in for free storage.
    *   Multiple notification channels.
    *   Support for multiple accounts (saves from the first account).

## Deployment

### Docker Deployment

Docker deployment offers a WebUI for managing configurations.  

```shell
docker run -d \
  --name quark-auto-save \
  -p 5005:5005 \ # Port mapping
  -e WEBUI_USERNAME=admin \
  -e WEBUI_PASSWORD=admin123 \
  -v ./quark-auto-save/config:/app/config \ # Configuration persistence
  -v ./quark-auto-save/media:/media \ # Optional, for alist_strm_gen strm usage
  --network bridge \
  --restart unless-stopped \
  cp0204/quark-auto-save:latest
  # registry.cn-shenzhen.aliyuncs.com/cp0204/quark-auto-save:latest # China mirror
```

**docker-compose.yml**

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

Access the WebUI at: `http://yourhost:5005`

| Environment Variable | Default     | Description                                 |
| -------------------- | ----------- | ------------------------------------------- |
| `WEBUI_USERNAME`     | `admin`     | WebUI username                              |
| `WEBUI_PASSWORD`     | `admin123`  | WebUI password                              |
| `PORT`               | `5005`      | WebUI port                                  |
| `PLUGIN_FLAGS`       |             | Disable plugins (e.g., `-emby,-aria2`)      |

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

### Regular Expression Examples

| Pattern                                | Replace                 | Result                                                                       |
| -------------------------------------- | ----------------------- | ---------------------------------------------------------------------------- |
| `.*`                                   |                         | Save all files without renaming.                                              |
| `\.mp4$`                               |                         | Save all `.mp4` files.                                                      |
| `^【电影TT】花好月圆(\d+)\.(mp4\|mkv)` | `\1.\2`                 | 【电影TT】花好月圆01.mp4 → 01.mp4<br>【电影TT】花好月圆02.mkv → 02.mkv     |
| `^(\d+)\.mp4`                          | `S02E\1.mp4`            | 01.mp4 → S02E01.mp4<br>02.mp4 → S02E02.mp4                                 |
| `$TV`                                  |                         | [Magic matching](#magic-matching) TV show files.                              |
| `^(\d+)\.mp4`                          | `{TASKNAME}.S02E\1.mp4` | 01.mp4 → TaskName.S02E01.mp4                                                   |

More regex details:  [Regex Tutorial](https://github.com/Cp0204/quark-auto-save/wiki/正则处理教程)

> [!TIP]
>
> **Magic Matching and Magic Variables:**  Use "magic matching" (regex starting with `$`) and "magic variables" (e.g., `{TASKNAME}`) for advanced renaming.
>
> See [Magic Matching and Magic Variables](https://github.com/Cp0204/quark-auto-save/wiki/魔法匹配和魔法变量)

### Refreshing Media Libraries

Configure plugins to automatically refresh your media library after saving files. [Plugin Configuration Guide](https://github.com/Cp0204/quark-auto-save/wiki/插件配置)

For custom integration, explore the [plugin development guide](https://github.com/Cp0204/quark-auto-save/tree/main/plugins).

### Additional Tips

Find more usage tips in the Wiki:  [Usage Tips](https://github.com/Cp0204/quark-auto-save/wiki/使用技巧集锦)

## Donations

If you find this project helpful, you can offer a small donation.

![WeChatPay](https://cdn.jsdelivr.net/gh/Cp0204/Cp0204@main/img/wechat_pay_qrcode.png)

## Disclaimer

This project is for personal use and aims to automate Quark Drive tasks.

The program does not involve any cracking and uses the official Quark API. I am not responsible for content or API changes. Use at your own discretion.

The project is open-source for learning and sharing. It is not for commercial use, nor is it authorized for commercial use. Illegal use is strictly prohibited.

## Sponsor

CDN acceleration and security protection for this project are sponsored by Tencent EdgeOne.

<a href="https://edgeone.ai/?from=github" target="_blank"><img title="Best Asian CDN, Edge, and Secure Solutions - Tencent EdgeOne" src="https://edgeone.ai/media/34fe3a45-492d-4ea4-ae5d-ea1087ca7b4b.png" width="300"></a>