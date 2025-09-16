<div align="center">
  <img src="img/icon.png" alt="Quark Auto Save Logo" width="100">
  <h1>Quark Auto Save: Automate Your Quark Cloud Drive Experience</h1>
  <p>Effortlessly automate Quark Cloud Drive tasks like auto-saving, renaming, and media library refresh with this powerful and easy-to-use tool.</p>
</div>

[![Wiki Documents](https://img.shields.io/badge/wiki-Documents-green?logo=github)](https://github.com/Cp0204/quark-auto-save/wiki)
[![GitHub Release](https://img.shields.io/github/v/release/Cp0204/quark-auto-save?logo=github)](https://github.com/Cp0204/quark-auto-save)
[![Docker Pulls](https://img.shields.io/docker/pulls/cp0204/quark-auto-save?logo=docker&logoColor=white)](https://hub.docker.com/r/cp0204/quark-auto-save)
[![Docker Image Size](https://img.shields.io/docker/image-size/cp0204/quark-auto-save?logo=docker&logoColor=white)](https://hub.docker.com/r/cp0204/quark-auto-save)

[**View the original repository on GitHub**](https://github.com/Cp0204/quark-auto-save)

![Run Log](img/run_log.png)

> [!CAUTION]
> ⛔️⛔️⛔️ **Important!** Resources are not updated constantly. **Avoid frequent scheduled runs** to prevent account risks and server strain.

> [!NOTE]
> The developer is not customer support. This is a free, open-source project. Please consult the Wiki and Issues before asking questions.

## Key Features

*   **Automated Saving:** Automatically transfer files from shared links to your Quark Cloud Drive.
*   **WebUI Management:** Configure and manage your tasks with an intuitive web interface via Docker.
*   **Flexible File Management:**  Rename files with regular expressions, skip already saved files, and ignore file extensions.
*   **Smart Link Handling:** Supports subdirectories in shared links and automatically handles links requiring passwords.
*   **Media Library Integration:** Integrates with media servers like Emby for automatic media library refreshing.
*   **Task Scheduling:**  Schedule tasks to run on specific days and set expiration dates.
*   **Notification Support:** Receive updates through various notification channels.
*   **Multi-Account Support:** Manage multiple accounts for increased flexibility.
*   **Daily Sign-in:** Automate the daily sign-in process to earn free cloud storage.
*   **Plugin Support:**  Extend functionality via plugin support.

## Deployment

### Docker Deployment

Docker deployment provides a user-friendly WebUI for easy configuration.

**Deployment Command:**

```shell
docker run -d \
  --name quark-auto-save \
  -p 5005:5005 \ # Change the port before the colon if needed; the port after the colon must remain unchanged
  -e WEBUI_USERNAME=admin \
  -e WEBUI_PASSWORD=admin123 \
  -v ./quark-auto-save/config:/app/config \ # Required: Persists configurations
  -v ./quark-auto-save/media:/media \ # Optional: For plugins like alist_strm_gen
  --network bridge \
  --restart unless-stopped \
  cp0204/quark-auto-save:latest
  # registry.cn-shenzhen.aliyuncs.com/cp0204/quark-auto-save:latest # China Mirror Address
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

**Management Address:** `http://yourhost:5005`

| Environment Variable | Default    | Notes                                      |
| -------------------- | ---------- | ------------------------------------------ |
| `WEBUI_USERNAME`     | `admin`    | Admin Username                             |
| `WEBUI_PASSWORD`     | `admin123` | Admin Password                             |
| `PORT`               | `5005`     | WebUI Port                                 |
| `PLUGIN_FLAGS`       |            | Disable specific plugins, e.g., `-emby,-aria2` |

#### One-Click Update

```shell
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock containrrr/watchtower -cR quark-auto-save
```

<details open>
<summary>WebUI Preview</summary>

![screenshot_webui](img/screenshot_webui-1.png)

![screenshot_webui](img/screenshot_webui-2.png)

</details>

## Usage & Configuration

### Regular Expression Examples

| Pattern                                | Replace                 | Result                                                                   |
| -------------------------------------- | ----------------------- | ---------------------------------------------------------------------- |
| `.*`                                   |                         | Save all files without any renaming.                                 |
| `\.mp4$`                               |                         | Save only files with the `.mp4` extension.                              |
| `^【电影TT】花好月圆(\d+)\.(mp4\|mkv)` | `\1.\2`                 | 【电影TT】花好月圆01.mp4 → 01.mp4<br>【电影TT】花好月圆02.mkv → 02.mkv |
| `^(\d+)\.mp4`                          | `S02E\1.mp4`            | 01.mp4 → S02E01.mp4<br>02.mp4 → S02E02.mp4                             |
| `$TV`                                  |                         | [Magic Match](#magic-match) episode file                                          |
| `^(\d+)\.mp4`                          | `{TASKNAME}.S02E\1.mp4` | 01.mp4 → 任务名.S02E01.mp4                                             |

More RegEx usage instructions: [RegEx Tutorial](https://github.com/Cp0204/quark-auto-save/wiki/正则处理教程)

> [!TIP]
>
> **Magic Matching and Magic Variables:** Use "magic matching" with patterns starting with `$` and an empty replacement.  Since v0.6.0, you can use more flexible "magic variables" enclosed in `{}`.
>
> See [Magic Matching and Magic Variables](https://github.com/Cp0204/quark-auto-save/wiki/魔法匹配和魔法变量) for more details.

### Refresh Media Library

Trigger actions after saving, like automatically refreshing the media library or generating .strm files.  Configuration: [Plugin Configuration](https://github.com/Cp0204/quark-auto-save/wiki/插件配置)

Media library integration is plugin-based.  See [Plugin Development Guide](https://github.com/Cp0204/quark-auto-save/tree/main/plugins) if you are interested in developing plugins.

### More Tips

See the Wiki: [Usage Tips](https://github.com/Cp0204/quark-auto-save/wiki/使用技巧集锦)

## Ecosystem Projects

Showcasing projects built around QAS, including official and third-party options.

### Official Projects

*   [QAS One-Click Push Assistant](https://greasyfork.org/zh-CN/scripts/533201-qas一键推送助手)

    A Greasemonkey script adds a "Push to QAS" button to Quark Cloud Drive sharing pages.

*   [SmartStrm](https://github.com/Cp0204/SmartStrm)

    .STRM file generation tool for post-processing and integration with media libraries.

### Third-Party Open Source Projects

> [!TIP]
>
> These third-party projects are community-developed and open-source.  Assess the risks before deploying to production.
>
> Submit new projects via Issues.

*   [nonebot-plugin-quark-autosave](https://github.com/fllesser/nonebot-plugin-quark-autosave)

    QAS Telegram bot for quick management of auto-saving tasks.

*   [Astrbot\_plugin\_quarksave](https://github.com/lm379/astrbot_plugin_quarksave)

    AstrBot plugin that uses quark\_auto\_save to automatically transfer resources to Quark Cloud Drive.

## Donations

If you find this project helpful, consider donating to show your support.

![WeChatPay](https://cdn.jsdelivr.net/gh/Cp0204/Cp0204@main/img/wechat_pay_qrcode.png)

## Disclaimer

This project is for personal use and aims to improve cloud drive efficiency.

The program doesn't use any cracking methods, and it relies on Quark's existing APIs. I am not responsible for the content on the drive or for any changes in the Quark API. Use at your own discretion.

Open source is for learning and communication. It is not for profit or commercial use.  Do not use it for illegal purposes.

## Sponsor

CDN acceleration and security protection for this project are sponsored by Tencent EdgeOne.

<a href="https://edgeone.ai/?from=github" target="_blank"><img title="Best Asian CDN, Edge, and Secure Solutions - Tencent EdgeOne" src="https://edgeone.ai/media/34fe3a45-492d-4ea4-ae5d-ea1087ca7b4b.png" width="300"></a>