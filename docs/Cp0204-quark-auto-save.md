<div align="center">
  <img src="img/icon.png" alt="quark-logo" width="100">
  <h1>Quark Auto-Save: Automate Your Quark Network Disk</h1>
  <p>Effortlessly automate Quark network disk tasks like auto-saving, renaming, and media library refreshing, saving you time and effort.  </p>

  [![Wiki][wiki-image]][wiki-url]
  [![GitHub Release][gitHub-releases-image]][github-url]
  [![Docker Pulls][docker-pulls-image]][docker-url]
  [![Docker Image Size][docker-image-size-image]][docker-url]

  [wiki-image]: https://img.shields.io/badge/wiki-Documents-green?logo=github
  [gitHub-releases-image]: https://img.shields.io/github/v/release/Cp0204/quark-auto-save?logo=github
  [docker-pulls-image]: https://img.shields.io/docker/pulls/cp0204/quark-auto-save?logo=docker&&logoColor=white
  [docker-image-size-image]: https://img.shields.io/docker/image-size/cp0204/quark-auto-save?logo=docker&&logoColor=white
  [github-url]: https://github.com/Cp0204/quark-auto-save
  [docker-url]: https://hub.docker.com/r/cp0204/quark-auto-save
  [wiki-url]: https://github.com/Cp0204/quark-auto-save/wiki

  <img src="img/run_log.png" alt="Run Log">
</div>

> [!CAUTION]
> ⚠️ **Important:** Avoid excessively frequent scheduled runs to prevent account risks and unnecessary server load.

> [!NOTE]
> The developer is not customer support. Please consult the Wiki and Issues before asking questions.

## Key Features

*   ✅ **Automated Quark Disk Tasks:** Automate Quark network disk tasks, including auto-saving, renaming, media library refreshing, and push notifications.
*   ✅ **Docker Deployment & WebUI:** Easy setup with Docker and a user-friendly WebUI for configuration.
*   ✅ **Share Link Management:**  Handles subdirectories, skips expired links, supports links with extraction codes, and intelligent resource search.
*   ✅ **File Management:** Creates target directories, skips already saved files, file name filtering with regular expressions, and renaming options.
*   ✅ **Task Management:** Supports multiple tasks with expiration dates and weekday-specific execution.
*   ✅ **Media Library Integration:**  Integrates with Emby to search and refresh media libraries automatically. Extensible with custom plugin modules.
*   ✅ **Additional Features:** Daily check-in for storage space, supports multiple notification channels, and multi-account support.

## Getting Started

### Docker Deployment

Deploy with Docker for a WebUI based management.

```shell
docker run -d \
  --name quark-auto-save \
  -p 5005:5005 \ # Port mapping, change the one before ":"
  -e WEBUI_USERNAME=admin \
  -e WEBUI_PASSWORD=admin123 \
  -v ./quark-auto-save/config:/app/config \ # Persistent configuration
  -v ./quark-auto-save/media:/media \ # Optional for alist_strm_gen
  --network bridge \
  --restart unless-stopped \
  cp0204/quark-auto-save:latest
  # registry.cn-shenzhen.aliyuncs.com/cp0204/quark-auto-save:latest # China mirror
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

Access the WebUI at: `http://yourhost:5005`

| Environment Variable  | Default    | Description                         |
| --------------------- | ---------- | ----------------------------------- |
| `WEBUI_USERNAME`    | `admin`    | Admin username                      |
| `WEBUI_PASSWORD`    | `admin123` | Admin password                      |
| `PORT`              | `5005`     | WebUI Port                          |
| `PLUGIN_FLAGS`      |            | Disable plugins using flags (e.g.,  `-emby,-aria2`) |

#### Update with Watchtower

```shell
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock containrrr/watchtower -cR quark-auto-save
```

<details open>
<summary>WebUI Preview</summary>

![screenshot_webui](img/screenshot_webui-1.png)
![screenshot_webui](img/screenshot_webui-2.png)

</details>

## Configuration & Usage

### Regular Expression Examples

| Pattern                                | Replace                 | Result                                                                               |
| -------------------------------------- | ----------------------- | ------------------------------------------------------------------------------------ |
| `.*`                                   |                         | Save all files without renaming                                                      |
| `\.mp4$`                               |                         | Save all files with the `.mp4` extension                                             |
| `^【电影TT】花好月圆(\d+)\.(mp4\|mkv)` | `\1.\2`                 | 【电影TT】花好月圆01.mp4 → 01.mp4<br>【电影TT】花好月圆02.mkv → 02.mkv             |
| `^(\d+)\.mp4`                          | `S02E\1.mp4`            | 01.mp4 → S02E01.mp4<br>02.mp4 → S02E02.mp4                                          |
| `$TV`                                  |                         | [Magic Matching](#magic-matching) for episodes                                     |
| `^(\d+)\.mp4`                          | `{TASKNAME}.S02E\1.mp4` | 01.mp4 → TaskName.S02E01.mp4                                                         |

More regex information: [Regex Tutorial](https://github.com/Cp0204/quark-auto-save/wiki/正则处理教程)

> [!TIP]
> **Magic Matching and Variables:** In regex processing, the "magic matching" mode starts with a `$` and is left empty in replace, it will be automatically replaced.  From v0.6.0, "magic variables" wrapped with `{}` can be used for more renaming.

For more details, refer to [Magic Matching and Variables](https://github.com/Cp0204/quark-auto-save/wiki/魔法匹配和魔法变量)

### Media Library Refresh

When new files are saved, it can trigger functions like media library refresh.  See: [Plugin Configuration](https://github.com/Cp0204/quark-auto-save/wiki/插件配置)

Media library modules are integrated via plugins.  See: [Plugin Development Guide](https://github.com/Cp0204/quark-auto-save/tree/main/plugins).

### Additional Tips

Consult the Wiki for more tips: [Usage Tips](https://github.com/Cp0204/quark-auto-save/wiki/使用技巧集锦)

## Support

If you found this project useful, consider a small donation. Thank you!

![WeChatPay](https://cdn.jsdelivr.net/gh/Cp0204/Cp0204@main/img/wechat_pay_qrcode.png)

## Disclaimer

This project is for personal use and to improve network disk efficiency through automation.

The program does not engage in any cracking behavior. It merely encapsulates existing APIs from Quark, and all data comes from the official Quark API. The author is not responsible for the content on the network disk nor for the impact of any changes in the official Quark APIs. Please use at your own discretion.

Open source for learning and communication only, is not for profit or for commercial use, and is strictly prohibited for illegal purposes.

## Sponsor

CDN acceleration and security protection for this project are sponsored by Tencent EdgeOne.

<a href="https://edgeone.ai/?from=github" target="_blank"><img title="Best Asian CDN, Edge, and Secure Solutions - Tencent EdgeOne" src="https://edgeone.ai/media/34fe3a45-492d-4ea4-ae5d-ea1087ca7b4b.png" width="300"></a>