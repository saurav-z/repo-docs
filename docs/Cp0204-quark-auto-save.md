<div align="center">
  <img src="img/icon.png" alt="Quark Auto-Save Logo" width="100">
</div>

# Quark Auto-Save: Automate Your Quark Cloud Drive Experience

**Effortlessly automate Quark Cloud Drive tasks, from auto-saving files to organizing your media library, with this powerful and easy-to-use tool. [Visit the original repository on GitHub](https://github.com/Cp0204/quark-auto-save).**

[![wiki][wiki-image]][wiki-url]
[![github releases][gitHub-releases-image]][github-url]
[![docker pulls][docker-pulls-image]][docker-url]
[![docker image size][docker-image-size-image]][docker-url]

[wiki-image]: https://img.shields.io/badge/wiki-Documents-green?logo=github
[gitHub-releases-image]: https://img.shields.io/github/v/release/Cp0204/quark-auto-save?logo=github
[docker-pulls-image]: https://img.shields.io/docker/pulls/cp0204/quark-auto-save?logo=docker&&logoColor=white
[docker-image-size-image]: https://img.shields.io/docker/image-size/cp0204/quark-auto-save?logo=docker&&logoColor=white
[github-url]: https://github.com/Cp0204/quark-auto-save
[docker-url]: https://hub.docker.com/r/cp0204/quark-auto-save
[wiki-url]: https://github.com/Cp0204/quark-auto-save/wiki

![run_log](img/run_log.png)

> [!CAUTION]
> ⛔️⛔️⛔️ **Important!**  Avoid excessive automated runs.  Limit the frequency to prevent account risks and reduce strain on the Quark servers.

> [!NOTE]
> The developer does not provide direct customer support.  The Wiki and Issues sections are the primary resources for troubleshooting. Please consult them before asking questions.

## Key Features

*   **Automated Transfers:** Automatically save files from shared links to your Quark Cloud Drive.
*   **Organized File Management:**  Organize files with custom renaming rules, skip already saved files, and filter by file extensions.
*   **Smart Linking:** Supports subdirectories within shared links and handles password-protected shares.
*   **Emby Integration:**  Automatically refreshes your Emby media library after new content is added (requires configuration).
*   **Flexible Task Management:** Manage multiple tasks, set expiration dates, and schedule tasks for specific days of the week.
*   **Daily Sign-In & Notifications:**  Automated daily sign-in to claim cloud storage and support for various notification channels.
*   **Multi-Account Support:** Manage multiple accounts (sign-in for all, auto-save for the primary account).
*   **Docker Deployment:** Easily deploy and manage the application via a user-friendly WebUI.

## Getting Started: Docker Deployment

Docker provides a simple and graphical way to configure the application.

```shell
docker run -d \
  --name quark-auto-save \
  -p 5005:5005 \ # Access via http://your_server_ip:5005, you can change the first port but not the second.
  -e WEBUI_USERNAME=admin \
  -e WEBUI_PASSWORD=admin123 \
  -v ./quark-auto-save/config:/app/config \ # Required: Configuration persistence
  -v ./quark-auto-save/media:/media \ # Optional: For strm generation with alist_strm_gen
  --network bridge \
  --restart unless-stopped \
  cp0204/quark-auto-save:latest
  # registry.cn-shenzhen.aliyuncs.com/cp0204/quark-auto-save:latest # Domestic mirror
```

Alternatively, use a `docker-compose.yml` file:

```yaml
version: "3.8"
services:
  quark-auto-save:
    image: cp0204/quark-auto-save:latest
    container_name: quark-auto-save
    network_mode: bridge
    ports:
      - "5005:5005"
    restart: unless-stopped
    environment:
      WEBUI_USERNAME: "admin"
      WEBUI_PASSWORD: "admin123"
    volumes:
      - ./quark-auto-save/config:/app/config
      - ./quark-auto-save/media:/media
```

Access the WebUI at:  `http://yourhost:5005`

| Environment Variable | Default   | Description                                |
| -------------------- | --------- | ------------------------------------------ |
| `WEBUI_USERNAME`     | `admin`   | WebUI login username                       |
| `WEBUI_PASSWORD`     | `admin123`| WebUI login password                       |
| `PORT`               | `5005`    | WebUI port                                 |
| `PLUGIN_FLAGS`       |           | Disable plugins (e.g., `-emby,-aria2`)     |

#### One-Click Update

```shell
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock containrrr/watchtower -cR quark-auto-save
```

<details open>
<summary>WebUI Preview</summary>

![screenshot_webui](img/screenshot_webui-1.png)

![screenshot_webui](img/screenshot_webui-2.png)

</details>

## File Renaming and Organization

Use regular expressions to customize how files are renamed and organized after they are saved.

### Regular Expression Examples

| Pattern                                  | Replace          | Result                                                                        |
| ---------------------------------------- | ---------------- | ----------------------------------------------------------------------------- |
| `.*`                                     |                  | Save all files without renaming.                                               |
| `\.mp4$`                                 |                  | Save only files with the .mp4 extension.                                      |
| `^【电影TT】花好月圆(\d+)\.(mp4\|mkv)`    | `\1.\2`          | 【电影TT】花好月圆01.mp4 → 01.mp4<br>【电影TT】花好月圆02.mkv → 02.mkv      |
| `^(\d+)\.mp4`                            | `S02E\1.mp4`     | 01.mp4 → S02E01.mp4<br>02.mp4 → S02E02.mp4                                 |
| `$TV`                                     |                  | [Magic Match](#magic-match) for episode files.                                 |
| `^(\d+)\.mp4`                            | `{TASKNAME}.S02E\1.mp4` | 01.mp4 → task_name.S02E01.mp4                                 |

For more regex examples: [Regex Tutorial](https://github.com/Cp0204/quark-auto-save/wiki/正则处理教程)

> [!TIP]
>
> **Magic Matching and Magic Variables:** The regex engine uses “magic matching” to simplify renaming tasks. Also supports {} "Magic Variables" since version v0.6.0.
>
> Learn more about magic matching: [Magic Match and Magic Variables](https://github.com/Cp0204/quark-auto-save/wiki/魔法匹配和魔法变量)

## Integrating with Media Libraries

Configure plugins to trigger actions like refreshing your media library (Emby) after new files are saved.  See the [Plugin Configuration Guide](https://github.com/Cp0204/quark-auto-save/wiki/插件配置).

If you wish to create your own media library plugin, check out the [plugin development guide](https://github.com/Cp0204/quark-auto-save/tree/main/plugins).

## Additional Resources

*   **Troubleshooting and Tips:** [Tips and Tricks](https://github.com/Cp0204/quark-auto-save/wiki/使用技巧集锦)
*   **FAQ:** [Wiki](https://github.com/Cp0204/quark-auto-save/wiki)

## Support the Project

If you find this project useful, you can show your appreciation by donating.

![WeChatPay](https://cdn.jsdelivr.net/gh/Cp0204/Cp0204@main/img/wechat_pay_qrcode.png)

## Disclaimer

This project is for personal use and automation purposes only.

It does not involve any cracking or unauthorized access to Quark's services. All data is obtained from the official Quark API.  The developer is not responsible for the content of the saved files, nor for any changes to the Quark API. Please use at your own discretion.

This is an open-source project for learning and discussion. It is not for commercial use, and any use for illegal purposes is strictly prohibited.

## Sponsor

CDN acceleration and security protection for this project are sponsored by Tencent EdgeOne.

<a href="https://edgeone.ai/?from=github" target="_blank"><img title="Best Asian CDN, Edge, and Secure Solutions - Tencent EdgeOne" src="https://edgeone.ai/media/34fe3a45-492d-4ea4-ae5d-ea1087ca7b4b.png" width="300"></a>
```
Key improvements and optimization notes:

*   **SEO Keywords:** Incorporated relevant keywords like "Quark Cloud Drive," "auto-save," "automation," "Docker," "Emby," "file organization," and "media library."
*   **Concise Hook:**  The one-sentence hook immediately highlights the core value proposition.
*   **Clear Headings:**  Used H2 headings for improved readability and structure.
*   **Bulleted Key Features:**  Features are presented in an easy-to-scan bulleted list.
*   **Actionable Instructions:** The Docker deployment section provides copy/paste instructions.
*   **Emphasis on Benefits:**  Highlights what users *gain* from using the software.
*   **Improved Readability:** Added line breaks and spacing for better readability.
*   **Clear Warnings:**  Warnings about overuse are emphasized at the beginning.
*   **Stronger Call to Action:** Encourages users to visit the GitHub repo.
*   **Structured Information:** Improved organization to enable faster reading.
*   **Optimized Disclaimer:** The disclaimer is clear and concise.