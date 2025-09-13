<div align="center">

![quark-logo](img/icon.png)

# Quark Auto-Save: Automate Your Quark Cloud Drive 🚀

**Automatically back up, organize, and refresh your Quark cloud drive with ease, streamlining your digital content management!**

[Visit the original repo on GitHub](https://github.com/Cp0204/quark-auto-save)

[![wiki][wiki-image]][wiki-url] [![github releases][gitHub-releases-image]][github-url] [![docker pulls][docker-pulls-image]][docker-url] [![docker image size][docker-image-size-image]][docker-url]

[wiki-image]: https://img.shields.io/badge/wiki-Documents-green?logo=github
[gitHub-releases-image]: https://img.shields.io/github/v/release/Cp0204/quark-auto-save?logo=github
[docker-url]: https://hub.docker.com/r/cp0204/quark-auto-save
[docker-pulls-image]: https://img.shields.io/docker/pulls/cp0204/quark-auto-save?logo=docker&&logoColor=white
[docker-image-size-image]: https://img.shields.io/docker/image-size/cp0204/quark-auto-save?logo=docker&&logoColor=white
[github-url]: https://github.com/Cp0204/quark-auto-save
[wiki-url]: https://github.com/Cp0204/quark-auto-save/wiki

![run_log](img/run_log.png)

</div>

> [!CAUTION]
> ⛔️⛔️⛔️ **Important!** Resources do not update constantly; **avoid overly frequent scheduling** to prevent account risks and server strain. Be mindful of resource limits.

> [!NOTE]
> The developer is not customer support. Open source is not a guarantee of solutions. Please consult the Wiki and Issues before asking questions.

## Key Features

*   **Automated Backups:** Automatically saves files to your Quark cloud drive.
*   **Docker Deployment:** Easy setup with a user-friendly WebUI for configuration.
*   **WebUI Configuration**:  Graphical configuration of all features
*   **Share Link Support:** Handles subdirectories and expired links, with support for links requiring passwords.
*   **File Management:** Automatically creates target directories, skips already saved files, and filters/renames files using regular expressions.
*   **Task Scheduling:** Manage multiple tasks, set end dates, and schedule tasks by day of the week.
*   **Media Library Integration:**  Integrates with Emby for automated library refreshing, allowing for seamless content updating.
*   **Notifications:** Supports various notification channels to keep you informed.
*   **Additional Features:** Includes daily sign-in for free space and multi-account support.

## Deployment

### Docker Deployment

Docker deployment provides a WebUI management interface for configuration, satisfying most needs graphically.

**Deployment Command:**

```shell
docker run -d \
  --name quark-auto-save \
  -p 5005:5005 \ # Adjust the port before the colon if needed (e.g., 8080:5005); the one after must remain unchanged.
  -e WEBUI_USERNAME=admin \
  -e WEBUI_PASSWORD=admin123 \
  -v ./quark-auto-save/config:/app/config \ # Required, for persistent configuration
  -v ./quark-auto-save/media:/media \ # Optional, for module alist_strm_gen to generate strm files
  --network bridge \
  --restart unless-stopped \
  cp0204/quark-auto-save:latest
  # registry.cn-shenzhen.aliyuncs.com/cp0204/quark-auto-save:latest # Domestic mirror address
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

**Management Address:** `http://yourhost:5005`

| Environment Variable | Default   | Notes                                     |
| -------------------- | --------- | ----------------------------------------- |
| `WEBUI_USERNAME`     | `admin`   | Admin Account                             |
| `WEBUI_PASSWORD`     | `admin123` | Admin Password                            |
| `PORT`               | `5005`    | Management Interface Port                 |
| `PLUGIN_FLAGS`       |           | Disable certain plugins, e.g., `-emby,-aria2` |

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

| Pattern                                | Replace                 | Effect                                                                    |
| -------------------------------------- | ----------------------- | ------------------------------------------------------------------------- |
| `.*`                                   |                         | Saves all files without any renaming.                                     |
| `\.mp4$`                               |                         | Saves all files with the .mp4 extension.                                  |
| `^【电影TT】花好月圆(\d+)\.(mp4\|mkv)` | `\1.\2`                 | 【电影TT】花好月圆01.mp4 → 01.mp4<br>【电影TT】花好月圆02.mkv → 02.mkv   |
| `^(\d+)\.mp4`                          | `S02E\1.mp4`            | 01.mp4 → S02E01.mp4<br>02.mp4 → S02E02.mp4                              |
| `$TV`                                  |                         | [Magic Match](#magic-match) handles episode files.                          |
| `^(\d+)\.mp4`                          | `{TASKNAME}.S02E\1.mp4` | 01.mp4 → Task Name.S02E01.mp4                                              |

For more information on Regular Expressions, see [Regular Expression Tutorial](https://github.com/Cp0204/quark-auto-save/wiki/Regular-Expression-Tutorial)

> [!TIP]
>
> **Magic Matching and Magic Variables**: In regular expression processing, we've defined some "magic match" patterns. If the value of an expression starts with `$`, and the replace field is left empty, the program will automatically use a predefined regular expression for matching and replacement.
>
> From v0.6.0 onwards, more "magic variables" enclosed in {} are supported for more flexible renaming.
>
> For more information, see [Magic Matching and Magic Variables](https://github.com/Cp0204/quark-auto-save/wiki/Magic-Matching-and-Magic-Variables)

### Refreshing Your Media Library

Upon new saves, trigger actions like automatically refreshing your media library or generating `.strm` files. Configuration guide: [Plugin Configuration](https://github.com/Cp0204/quark-auto-save/wiki/Plugin-Configuration)

Media library modules are integrated as plugins. If you're interested, refer to the [Plugin Development Guide](https://github.com/Cp0204/quark-auto-save/tree/main/plugins).

### More Tips and Tricks

Consult the Wiki: [Tips and Tricks](https://github.com/Cp0204/quark-auto-save/wiki/Tips-and-Tricks)

## Donation

If this project has benefited you, consider donating to show your appreciation. Thank you!

![WeChatPay](https://cdn.jsdelivr.net/gh/Cp0204/Cp0204@main/img/wechat_pay_qrcode.png)

## Disclaimer

This project is developed for personal interest, designed to improve cloud drive efficiency through automation.

The program does not engage in any cracking activities; it merely encapsulates the Quark API. All data comes from the official Quark API. The author is not responsible for the content stored, nor for any impact caused by potential changes to the official Quark API. Please use at your own discretion.

Open source is intended for learning and communication purposes. It is not for profit or authorized for commercial use. Strict prohibition of illegal usage.

## Sponsor

CDN acceleration and security protection for this project are sponsored by Tencent EdgeOne.

<a href="https://edgeone.ai/?from=github" target="_blank"><img title="Best Asian CDN, Edge, and Secure Solutions - Tencent EdgeOne" src="https://edgeone.ai/media/34fe3a45-492d-4ea4-ae5d-ea1087ca7b4b.png" width="300"></a>
```
Key improvements and explanations:

*   **SEO Optimization:**  Incorporated keywords like "Quark Auto-Save," "cloud drive," "automation," and "backup."  Headings and subheadings structure the content for readability and SEO.
*   **Concise Hook:** A compelling one-sentence opening to immediately capture the user's attention.
*   **Clear Structure:** Uses headings, bullet points, and tables to organize information and make it easier to scan.
*   **Detailed Feature Descriptions:**  Provides more descriptive text for key features, highlighting their value.
*   **Emphasis on User Benefits:** The text focuses on what the tool *does* for the user (e.g., "Automated Backups," "Media Library Integration").
*   **Docker Deployment Enhanced:** The Docker section is significantly improved, with clear instructions and explanations.  Includes the `docker-compose.yml` file.  Added a One-click update section.
*   **Regular Expression Clarity:** Improved the Regular Expression section with more examples and explanations.
*   **Call to Action:** Includes a clear call to action to visit the original repo.
*   **Concise Disclaimer:** The disclaimer is kept short and to the point.
*   **Sponsor Section:** The sponsor information is included as provided.
*   **Removed Redundancy and Unnecessary Details:**  Removed some of the very specific details in the original readme, such as the history of the project, to avoid clutter.
*   **Added missing "Note" and "Tip" boxes:** Added missing note/tip boxes to better format the document.
*   **Added links:** Hyperlinked wiki pages.

This improved README is more informative, easier to read, and much better optimized for both users and search engines.  It is also a more complete and functional introduction to the project.