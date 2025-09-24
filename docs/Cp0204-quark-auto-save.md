<div align="center">
  <img src="img/icon.png" alt="Quark Auto Save Logo" width="100">

  # Quark Auto Save: Automate Your Quark Network Disk (夸克网盘) Experience

  **Effortlessly automate Quark Network Disk tasks with Quark Auto Save, saving, organizing, and refreshing your media library.**

  [![Wiki][wiki-image]][wiki-url] [![GitHub Releases][gitHub-releases-image]][github-url] [![Docker Pulls][docker-pulls-image]][docker-url] [![Docker Image Size][docker-image-size-image]][docker-url]

  [wiki-image]: https://img.shields.io/badge/wiki-Documents-green?logo=github
  [gitHub-releases-image]: https://img.shields.io/github/v/release/Cp0204/quark-auto-save?logo=github
  [docker-pulls-image]: https://img.shields.io/docker/pulls/cp0204/quark-auto-save?logo=docker&&logoColor=white
  [docker-image-size-image]: https://img.shields.io/docker/image-size/cp0204/quark-auto-save?logo=docker&&logoColor=white
  [github-url]: https://github.com/Cp0204/quark-auto-save
  [docker-url]: https://hub.docker.com/r/cp0204/quark-auto-save
  [wiki-url]: https://github.com/Cp0204/quark-auto-save/wiki

  ![Run Log Example](img/run_log.png)
</div>

> [!CAUTION]
> ⛔️⛔️⛔️ **Important!**  Resources do not update constantly; **avoid excessively frequent scheduling** to prevent account risks and server load. Every action has consequences!

> [!NOTE]
> The developer is *not* customer support. Open source and free do not equal support for usage issues. Please consult the Wiki and Issues before asking questions; the Wiki is comprehensive.

## Key Features

*   **Automated Saving:** Automatically transfers files from shared links to your Quark Network Disk.
*   **WebUI Management:** Easy-to-use web interface for configuration and management via Docker.
*   **File Organization:** Organize files with automatic renaming and filtering, plus support for subdirectories and skipped files.
*   **Media Library Integration:** Integrates with media servers like Emby for automatic library updates.
*   **Task Management:** Supports multiple tasks with flexible scheduling and expiration dates.
*   **Notifications:** Receive updates via multiple notification channels.
*   **Account Management:** Supports multiple Quark accounts.
*   **Share Link Support:** Works with share links, including those with passwords.
*   **Emby Integration:** Automatically refreshes Emby media libraries after transfers and rename.
*   **Sign-in Rewards:** Automatically signs into Quark Network Disk to earn space.

## Deployment

### Docker Deployment

Docker provides a WebUI for easy configuration.

**Simple Deployment Command:**

```bash
docker run -d \
  --name quark-auto-save \
  -p 5005:5005 \ # Access port (adjust the port before the colon)
  -e WEBUI_USERNAME=admin \
  -e WEBUI_PASSWORD=admin123 \
  -v ./quark-auto-save/config:/app/config \ # Configuration persistence
  -v ./quark-auto-save/media:/media \ # Optional: for alist_strm_gen
  --network bridge \
  --restart unless-stopped \
  cp0204/quark-auto-save:latest
  # registry.cn-shenzhen.aliyuncs.com/cp0204/quark-auto-save:latest # China mirror
```

**docker-compose.yml**

```yaml
version: "3.8"  # Or your preferred version
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

| Environment Variable | Default      | Notes                                      |
| -------------------- | ------------ | ------------------------------------------ |
| `WEBUI_USERNAME`     | `admin`      | WebUI login username                       |
| `WEBUI_PASSWORD`     | `admin123`   | WebUI login password                       |
| `PORT`               | `5005`       | WebUI access port                          |
| `PLUGIN_FLAGS`       |              | Disable specific plugins, e.g., `-emby,-aria2` |

#### One-Click Updates

```bash
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock containrrr/watchtower -cR quark-auto-save
```

<details open>
<summary>WebUI Preview</summary>

![screenshot_webui](img/screenshot_webui-1.png)

![screenshot_webui](img/screenshot_webui-2.png)

</details>

## Usage

### Regular Expression Examples

| Pattern                                | Replace                 | Result                                                              |
| -------------------------------------- | ----------------------- | ------------------------------------------------------------------- |
| `.*`                                   |                         | Transfer all files without any renaming.                             |
| `\.mp4$`                               |                         | Transfer only `.mp4` files.                                         |
| `^【MovieTT】FlowerMoon(\d+)\.(mp4\|mkv)` | `\1.\2`                 | 【MovieTT】FlowerMoon01.mp4 → 01.mp4<br>【MovieTT】FlowerMoon02.mkv → 02.mkv |
| `^(\d+)\.mp4`                          | `S02E\1.mp4`            | 01.mp4 → S02E01.mp4<br>02.mp4 → S02E02.mp4                             |
| `$TV`                                  |                         | [Magic Match](#魔法匹配) for TV series files.                      |
| `^(\d+)\.mp4`                          | `{TASKNAME}.S02E\1.mp4` | 01.mp4 → TaskName.S02E01.mp4                                        |

More regex usage instructions: [Regex Processing Tutorial](https://github.com/Cp0204/quark-auto-save/wiki/正则处理教程)

> [!TIP]
>
> **Magic Matching and Magic Variables:** Special patterns are available within the regex processing. If the expression value starts with `$` and the replace value is empty, the program will use predefined regex patterns.
>
> Support for "magic variables" enclosed in `{}` for more flexible renaming is available from v0.6.0 onwards.
>
> See [Magic Matching and Magic Variables](https://github.com/Cp0204/quark-auto-save/wiki/魔法匹配和魔法变量) for details.

### Media Library Refresh

Trigger actions like refreshing your media library or generating `.strm` files when new transfers occur. Configuration guide: [Plugin Configuration](https://github.com/Cp0204/quark-auto-save/wiki/插件配置)

Media library modules are integrated as plugins. For plugin development, see the [Plugin Development Guide](https://github.com/Cp0204/quark-auto-save/tree/main/plugins).

### More Tips

Refer to the Wiki for more usage tips: [Tips and Tricks](https://github.com/Cp0204/quark-auto-save/wiki/使用技巧集锦)

## Ecosystem Projects

*   [QAS One-Click Push Helper](https://greasyfork.org/zh-CN/scripts/533201-qas一键推送助手) - Greasemonkey script to add a "Push to QAS" button to Quark Network Disk share pages.
*   [SmartStrm](https://github.com/Cp0204/SmartStrm) - STRM file generator for post-transfer processing, enabling media playback without downloading.

### Third-Party Open Source Projects

> [!TIP]
>
> These third-party projects are community-developed and open source, not directly affiliated with the QAS author.  Assess risks before deployment.
>
> Submit new projects via Issues.

*   [nonebot-plugin-quark-autosave](https://github.com/fllesser/nonebot-plugin-quark-autosave) - A QAS Telegram bot for easy management of auto-transfer tasks.
*   [Astrbot_plugin_quarksave](https://github.com/lm379/astrbot_plugin_quarksave) - AstrBot plugin that calls quark_auto_save to automatically transfer resources to Quark Network Disk.

## Donations

If you find this project helpful, consider donating to support its development.

![WeChatPay](https://cdn.jsdelivr.net/gh/Cp0204/Cp0204@main/img/wechat_pay_qrcode.png)

## Disclaimer

This project is for personal use, aimed at automating and improving Quark Network Disk usage.

It doesn't involve any cracking and only utilizes the official Quark API. I am not responsible for content on the disks, or any changes to the Quark API that may affect this program.  Use at your own risk.

This is open source for learning and exchange purposes only.  It's not for commercial use.  Illegal use is strictly prohibited.

## Sponsor

CDN acceleration and security protection for this project are sponsored by Tencent EdgeOne.

<a href="https://edgeone.ai/?from=github" target="_blank"><img title="Best Asian CDN, Edge, and Secure Solutions - Tencent EdgeOne" src="https://edgeone.ai/media/34fe3a45-492d-4ea4-ae5d-ea1087ca7b4b.png" width="300"></a>
```
Key improvements and explanations:

*   **SEO-Friendly Title and Hook:**  The title and the one-sentence hook are keyword-rich ("Quark Auto Save," "Automate," "Quark Network Disk," "saving, organizing, refreshing").  The hook immediately states the benefit.
*   **Clear Headings:**  Uses clear, descriptive headings for each section (Key Features, Deployment, Usage, etc.).
*   **Bulleted Key Features:** Uses a concise bulleted list for easy readability. This highlights the core functionality.
*   **Concise Language:**  Streamlined the language throughout for better clarity and impact.
*   **Emphasis on Benefits:**  The "Key Features" section focuses on the benefits users receive (automation, organization, media library integration).
*   **Actionable Deployment Instructions:**  The Docker deployment section includes clear instructions, commands, and environment variable explanations, along with a `docker-compose` example.
*   **Links Back to Repo:** Keeps all original links.
*   **Added Ecosystem and Third-Party Project Sections:** This highlights that the project integrates into a wider community and emphasizes the project's usefulness.
*   **Updated Disclaimer and Sponsor Information**
*   **More SEO friendly keywords in headings**

This improved README is more informative, easier to understand, and more likely to attract users and improve search engine ranking.