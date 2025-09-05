<div align="center">
  <img src="img/icon.png" alt="Quark Auto-Save Logo" width="100">
  <h1>Quark Auto-Save: Automate Your Quark Drive!</h1>
</div>

Tired of manually saving files to your Quark Drive? **Quark Auto-Save automates Quark Drive tasks like auto-saving, renaming, and media library refreshing, all in one convenient package!**  ([Back to Original Repo](https://github.com/Cp0204/quark-auto-save))

[![Wiki][wiki-image]][wiki-url]
[![GitHub Releases][gitHub-releases-image]][github-url]
[![Docker Pulls][docker-pulls-image]][docker-url]
[![Docker Image Size][docker-image-size-image]][docker-url]

[wiki-image]: https://img.shields.io/badge/wiki-Documents-green?logo=github
[gitHub-releases-image]: https://img.shields.io/github/v/release/Cp0204/quark-auto-save?logo=github
[docker-pulls-image]: https://img.shields.io/docker/pulls/cp0204/quark-auto-save?logo=docker&&logoColor=white
[docker-image-size-image]: https://img.shields.io/docker/image-size/cp0204/quark-auto-save?logo=docker&&logoColor=white
[github-url]: https://github.com/Cp0204/quark-auto-save
[docker-url]: https://hub.docker.com/r/cp0204/quark-auto-save
[wiki-url]: https://github.com/Cp0204/quark-auto-save/wiki

<img src="img/run_log.png" alt="Run Log Screenshot">

> [!IMPORTANT]
> ⚠️ **Avoid Frequent Runs:**  To protect your account and minimize server load, avoid setting excessively frequent automated runs.

> [!NOTE]
> **Support:**  Please consult the Wiki and existing Issues before asking for help. The Wiki has comprehensive documentation.

## Key Features

*   ✅ **Automated Quark Drive Tasks:** Automate the process of transferring, renaming, and organizing files within your Quark Drive.
*   ✅ **Docker Deployment:** Easy deployment with a user-friendly WebUI for configuration.
*   ✅ **Share Link Support:** Supports subdirectories within share links, handles expired links, and extracts passwords.
*   ✅ **Smart File Handling:** Automatically skips already saved files, and intelligently handles file name filtering and renaming using regular expressions.
*   ✅ **Flexible Task Management:** Create and manage multiple tasks with options for scheduled execution, including specifying days of the week.
*   ✅ **Media Library Integration:** Automatically refreshes media libraries (e.g., Emby) after transfers, and supports custom hooks for advanced integration.
*   ✅ **Additional Functionality:** Includes daily sign-in for bonus storage, supports multiple notification channels, and allows for multi-account sign-in.

## Deployment

### Docker Deployment

Docker deployment offers a WebUI for easy configuration.  A Docker container allows you to graphically configure the majority of functionality.

```bash
docker run -d \
  --name quark-auto-save \
  -p 5005:5005 \ # Map port, you can change the one before the colon (:) but not after
  -e WEBUI_USERNAME=admin \
  -e WEBUI_PASSWORD=admin123 \
  -v ./quark-auto-save/config:/app/config \ # Required: Configuration persistence
  -v ./quark-auto-save/media:/media \ # Optional: For alist_strm_gen module
  --network bridge \
  --restart unless-stopped \
  cp0204/quark-auto-save:latest
  # registry.cn-shenzhen.aliyuncs.com/cp0204/quark-auto-save:latest # China Mirror
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

Access the WebUI at: `http://yourhost:5005`

| Environment Variable | Default     | Description                         |
| -------------------- | ----------- | ----------------------------------- |
| `WEBUI_USERNAME`     | `admin`     | Admin Username                      |
| `WEBUI_PASSWORD`     | `admin123`  | Admin Password                      |
| `PORT`               | `5005`      | WebUI Port                          |
| `PLUGIN_FLAGS`       |             | Disable plugins, e.g., `-emby,-aria2` |

#### One-Click Update

```bash
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock containrrr/watchtower -cR quark-auto-save
```

<details>
<summary>WebUI Preview</summary>

<img src="img/screenshot_webui-1.png" alt="WebUI Screenshot 1">
<img src="img/screenshot_webui-2.png" alt="WebUI Screenshot 2">

</details>

## Usage & Configuration

### Regular Expression Examples

| Pattern                                  | Replace                 | Result                                                                   |
| ---------------------------------------- | ----------------------- | ---------------------------------------------------------------------- |
| `.*`                                     |                         | Transfer all files, no renaming.                                       |
| `\.mp4$`                                 |                         | Transfer all files with the `.mp4` extension.                             |
| `^【MovieTT】FlowerMoon(\d+)\.(mp4\|mkv)` | `\1.\2`                 | 【MovieTT】FlowerMoon01.mp4 -> 01.mp4<br>【MovieTT】FlowerMoon02.mkv -> 02.mkv |
| `^(\d+)\.mp4`                            | `S02E\1.mp4`            | 01.mp4 -> S02E01.mp4<br>02.mp4 -> S02E02.mp4                             |
| `$TV`                                     |                         | [Magic Match](#magic-match) for TV series files                             |
| `^(\d+)\.mp4`                            | `{TASKNAME}.S02E\1.mp4` | 01.mp4 -> task_name.S02E01.mp4                                        |

More regex usage instructions: [Regex Tutorial](https://github.com/Cp0204/quark-auto-save/wiki/Regex-Tutorial)

> [!TIP]
>
> **Magic Matching and Magic Variables:**  We define "magic match" patterns in regex. If the expression starts with `$`, and the replacement is blank, the program uses pre-defined regex for matching and replacement.
>
> Since v0.6.0, it supports more "magic variables" enclosed in `{}`, which provide more flexible renaming.
>
> See the  [Magic Matching and Magic Variables](https://github.com/Cp0204/quark-auto-save/wiki/Magic-Matching-and-Magic-Variables) for more.

### Media Library Refresh

After new transfers, the tool can trigger actions like refreshing your media library or generating .strm files. Configuration guide: [Plugin Configuration](https://github.com/Cp0204/quark-auto-save/wiki/Plugin-Configuration)

Media library modules are integrated as plugins.  If you're interested, see the [Plugin Development Guide](https://github.com/Cp0204/quark-auto-save/tree/main/plugins).

### More Tips & Tricks

Refer to the Wiki: [Tips & Tricks](https://github.com/Cp0204/quark-auto-save/wiki/Tips-and-Tricks)

## Donations

If this project is helpful, you can donate to support its development.

![WeChatPay](https://cdn.jsdelivr.net/gh/Cp0204/Cp0204@main/img/wechat_pay_qrcode.png)

## Disclaimer

This project is developed for personal use and aims to improve Quark Drive efficiency through automation.

The program does not involve any cracking and only uses Quark's existing APIs. The author is not responsible for the content on the drive, or for any changes to the Quark API. Use this at your own discretion.

This is open source for learning and communication purposes only, is not for profit, and is not authorized for commercial use.  It is strictly prohibited for illegal purposes.

## Sponsor

CDN acceleration and security protection for this project are sponsored by Tencent EdgeOne.

<a href="https://edgeone.ai/?from=github" target="_blank"><img title="Best Asian CDN, Edge, and Secure Solutions - Tencent EdgeOne" src="https://edgeone.ai/media/34fe3a45-492d-ea4-ae5d-ea1087ca7b4b.png" width="300"></a>
```
Key improvements and summaries:

*   **SEO Optimization:**  Includes the project name, relevant keywords (Quark Drive, automation, auto-save), and more specific descriptions in the introduction and feature bullets.  Uses headers to structure the content for easier scanning.
*   **Concise Hook:**  Starts with a strong one-sentence description of the project's purpose.
*   **Clear Headings and Structure:**  Uses clear headings and subheadings to improve readability and navigation.
*   **Feature Summary with Bullets:**  Highlights the core features with clear, concise bullet points.
*   **Docker Deployment Instructions:** Simplified and directly accessible Docker deployment instructions.
*   **Removed Irrelevant or Redundant Text:** Streamlined the README to focus on the most important information.
*   **Improved Formatting:** Enhanced formatting for better visual appeal and readability, using Markdown effectively.
*   **Emphasis on Key Sections:**  Highlights important information like the caution about usage frequency and the disclaimer.
*   **Wiki and Tip Links:**  Kept links to useful information.
*   **Call to Action:** Includes a clear "back to repo" link.