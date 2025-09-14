<div align="center">

![quark-logo](img/icon.png)

# Quark Auto-Save: Automate Your Quark Cloud Drive

**Automatically save, organize, and manage your Quark cloud drive content with Quark Auto-Save, making it easy to keep your files updated.**

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
> ⛔️⛔️⛔️ **Important!** Resources are not updated constantly; **do not set overly frequent scheduled runs** to avoid account risks and unnecessary strain on the Quark servers. Every action has its consequence.

> [!NOTE]
> The developer is not customer support. Open source is free, but doesn't guarantee problem resolution; please refer to the Wiki and Issues first, before asking questions.

## Key Features

*   **Automated Saving:** Automatically transfers files from share links to your Quark drive.
*   **WebUI & Docker Deployment:** Easy setup and configuration via a user-friendly web interface.
*   **Filename Management:** Organize your files with renaming rules using regex.
*   **Media Library Integration:** Refreshes your media library (e.g., Emby) after new content is saved.
*   **Share Link Support:** Handles subdirectories and supports share links requiring passwords.
*   **Task Scheduling:** Schedule tasks to run on specific days and set expiration dates.
*   **Emby Integration**: Search Emby media library based on the task name.
*   **Notifications:** Supports multiple notification channels.
*   **Daily Sign-in:** Get space by daily signing-in.
*   **Multi-Account Support:** Supports multiple accounts, signing in all of them and transferring resources in one account only.

## Deployment

### Docker Deployment

Docker deployment offers a WebUI for management and configuration, which meets most requirements.

Run the following command:

```shell
docker run -d \
  --name quark-auto-save \
  -p 5005:5005 \ # Map the port, the one before ":" can be changed, which is the port accessed after deployment, the one after ":" cannot be changed
  -e WEBUI_USERNAME=admin \
  -e WEBUI_PASSWORD=admin123 \
  -v ./quark-auto-save/config:/app/config \ # Required, configuration persistence
  -v ./quark-auto-save/media:/media \ # Optional, used by the module alist_strm_gen for generating strm
  --network bridge \
  --restart unless-stopped \
  cp0204/quark-auto-save:latest
  # registry.cn-shenzhen.aliyuncs.com/cp0204/quark-auto-save:latest # Domestic mirror address
```

Alternatively, use docker-compose.yml:

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

| Environment Variable | Default     | Notes                               |
| -------------------- | ----------- | ----------------------------------- |
| `WEBUI_USERNAME`     | `admin`     | Management account                  |
| `WEBUI_PASSWORD`     | `admin123`  | Management password                 |
| `PORT`               | `5005`      | Management backend port             |
| `PLUGIN_FLAGS`       |             | Plugin flags, such as `-emby,-aria2` |

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

### Regex Processing Examples

| Pattern                                | Replace                 | Effect                                                                        |
| -------------------------------------- | ----------------------- | ----------------------------------------------------------------------------- |
| `.*`                                   |                         | Save all files without any organization                                       |
| `\.mp4$`                               |                         | Save all files with the `.mp4` suffix                                         |
| `^【电影TT】花好月圆(\d+)\.(mp4\|mkv)` | `\1.\2`                 | 【电影TT】花好月圆01.mp4 → 01.mp4<br>【电影TT】花好月圆02.mkv → 02.mkv           |
| `^(\d+)\.mp4`                          | `S02E\1.mp4`            | 01.mp4 → S02E01.mp4<br>02.mp4 → S02E02.mp4                                   |
| `$TV`                                  |                         | [Magic Matching](#magic-matching) series files                                |
| `^(\d+)\.mp4`                          | `{TASKNAME}.S02E\1.mp4` | 01.mp4 → task name.S02E01.mp4                                                |

More regex usage instructions: [Regex Processing Tutorial](https://github.com/Cp0204/quark-auto-save/wiki/Regex-Processing-Tutorial)

> [!TIP]
>
> **Magic Matching and Magic Variables:** During regex processing, we define some "magic matching" patterns. If the expression value starts with `$`, and the replacement field is left blank, the program will automatically use the preset regular expression for matching and replacing.
>
> Starting from v0.6.0, it supports more "magic variables" enclosed in `{}` which I call, allowing for more flexible renaming.
>
> For more information, please refer to [Magic Matching and Magic Variables](https://github.com/Cp0204/quark-auto-save/wiki/Magic-Matching-and-Magic-Variables)

### Refreshing the Media Library

When new content is saved, you can trigger functions such as automatically refreshing the media library or generating .strm files. Configuration guide: [Plugin Configuration](https://github.com/Cp0204/quark-auto-save/wiki/Plugin-Configuration)

Media library modules are integrated as plugins. If you're interested, please refer to the [Plugin Development Guide](https://github.com/Cp0204/quark-auto-save/tree/main/plugins).

### More Usage Tips

Please refer to the Wiki: [Usage Tips Collection](https://github.com/Cp0204/quark-auto-save/wiki/Usage-Tips-Collection)

## Donate

If this project benefits you, you can donate. Thank you!

![WeChatPay](https://cdn.jsdelivr.net/gh/Cp0204/Cp0204@main/img/wechat_pay_qrcode.png)

## Disclaimer

This project is developed out of personal interest to improve cloud drive usage efficiency through automation.

The program does not involve any cracking behaviors; it only encapsulates existing APIs of Quark. All data comes from Quark's official APIs. The author is not responsible for the content on the cloud drive, or for any impact caused by potential changes in Quark's official APIs in the future; please consider your usage carefully.

Open source is for learning and communication purposes only, is not for profit or authorized commercial use, and is strictly prohibited for illegal purposes.

## Sponsor

CDN acceleration and security protection for this project are sponsored by Tencent EdgeOne.

<a href="https://edgeone.ai/?from=github" target="_blank"><img title="Best Asian CDN, Edge, and Secure Solutions - Tencent EdgeOne" src="https://edgeone.ai/media/34fe3a45-492d-4ea4-ae5d-ea1087ca7b4b.png" width="300"></a>

```

Key improvements and SEO optimizations:

*   **Concise Hook:** The opening sentence is now a clear value proposition.
*   **Targeted Keywords:** Includes relevant keywords like "Quark Auto-Save," "cloud drive," "automation," "file management," and "media library."
*   **Clear Headings:** Uses H2 and H3 for better structure and readability.
*   **Bulleted Key Features:** Highlights the main benefits in an easy-to-scan format.
*   **Strong Call to Action:** The hook and key features draw the user in.
*   **Docker Emphasis:** Because that's a key part of the offering, it is prominently highlighted.
*   **SEO-Friendly Formatting:** Uses markdown for clean structure.
*   **Concise Language:** Avoids unnecessary words.
*   **Direct Links:** Uses the original repo link.
*   **Clear Disclaimer:** Retains the important disclaimer.