<div align="center">

![quark-logo](img/icon.png)

# Quark Auto-Save: Automate Your Quark Cloud Drive with Ease

**Automatically save, organize, and manage your files in Quark Cloud Drive with this powerful and user-friendly automation tool!** 

Check out the original repo: [https://github.com/Cp0204/quark-auto-save](https://github.com/Cp0204/quark-auto-save)

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
> ⛔️⛔️⛔️ **Important:** Avoid excessive scheduling to prevent account risks and unnecessary load on Quark servers.

> [!NOTE]
> The developer is not a customer service representative. This is an open-source project; assistance with usage issues is provided through the Wiki and Issues, and is not guaranteed.

## Key Features

*   **Automated Saving:** Automatically transfers files from shared links to your Quark Cloud Drive.
*   **WebUI Configuration:**  Easy-to-use WebUI for managing your automation tasks via Docker deployment.
*   **Smart Link Handling:** Supports subdirectories within shared links and handles links requiring extraction codes.
*   **File Management:** Automates the creation of target directories, skips already saved files, and filters/renames files using regular expressions.
*   **Flexible Task Management:** Supports multiple tasks with individual scheduling options and end dates.
*   **Media Library Integration:** Seamlessly integrates with media libraries (Emby) for automatic updates and library refreshing.
*   **Additional Functionality:** Includes daily check-in for extra storage and supports multiple notification channels and accounts.

## Installation and Usage

### Docker Deployment (Recommended)

Deploying via Docker provides a WebUI for easy configuration.

**Docker Run Command:**

```shell
docker run -d \
  --name quark-auto-save \
  -p 5005:5005 \ # Customize the port before the colon
  -e WEBUI_USERNAME=admin \
  -e WEBUI_PASSWORD=admin123 \
  -v ./quark-auto-save/config:/app/config \ # Configuration persistence
  -v ./quark-auto-save/media:/media \ # Optional: For alist_strm_gen plugin
  --network bridge \
  --restart unless-stopped \
  cp0204/quark-auto-save:latest
  # registry.cn-shenzhen.aliyuncs.com/cp0204/quark-auto-save:latest # China Mirror
```

**docker-compose.yml Example:**

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

**Access the WebUI:** `http://yourhost:5005`

| Environment Variable | Default      | Description                                 |
| -------------------- | ------------ | ------------------------------------------- |
| `WEBUI_USERNAME`     | `admin`      | WebUI Username                              |
| `WEBUI_PASSWORD`     | `admin123`   | WebUI Password                              |
| `PORT`               | `5005`       | WebUI Port                                  |
| `PLUGIN_FLAGS`       |              | Disable plugins using flags such as `-emby`. |

#### One-Click Update

```shell
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock containrrr/watchtower -cR quark-auto-save
```

<details open>
<summary>WebUI Preview</summary>

![screenshot_webui](img/screenshot_webui-1.png)

![screenshot_webui](img/screenshot_webui-2.png)

</details>

### Regular Expression Examples for File Management

| Pattern                                | Replace                 | Result                                                                   |
| -------------------------------------- | ----------------------- | ---------------------------------------------------------------------- |
| `.*`                                   |                         | Saves all files without any changes.                                               |
| `\.mp4$`                               |                         | Saves all files with the `.mp4` extension.                                             |
| `^【电影TT】花好月圆(\d+)\.(mp4\|mkv)` | `\1.\2`                 | 【电影TT】花好月圆01.mp4 → 01.mp4<br>【电影TT】花好月圆02.mkv → 02.mkv |
| `^(\d+)\.mp4`                          | `S02E\1.mp4`            | 01.mp4 → S02E01.mp4<br>02.mp4 → S02E02.mp4                             |
| `$TV`                                  |                         | [Magic Match](#魔法匹配) for episode files.                                          |
| `^(\d+)\.mp4`                          | `{TASKNAME}.S02E\1.mp4` | 01.mp4 → 任务名.S02E01.mp4                                             |

More examples: [Regex Processing Tutorial](https://github.com/Cp0204/quark-auto-save/wiki/正则处理教程)

> [!TIP]
> **Magic Matching and Magic Variables:** These provide advanced renaming using predefined regex patterns and custom variables. 
>
> See the [Magic Matching and Magic Variables](https://github.com/Cp0204/quark-auto-save/wiki/魔法匹配和魔法变量) documentation for details.

### Media Library Refresh

The plugin architecture allows the program to interact with media servers, trigger actions such as refreshing the media library, or generating .strm files after saving new files.  Find out more at [Plugin Configuration](https://github.com/Cp0204/quark-auto-save/wiki/插件配置)

Interested in building your own media library plugin? Check out the [plugin development guide](./plugins).

### Additional Usage Tips

For further guidance, consult the Wiki: [Usage Tips](https://github.com/Cp0204/quark-auto-save/wiki/使用技巧集锦)

## Ecosystem Projects

Showcasing projects that extend the functionality of QAS.

### Official Projects

*   [QAS One-Click Push Assistant](https://greasyfork.org/zh-CN/scripts/533201-qas一键推送助手)

    A userscript for adding a "Push to QAS" button on Quark Cloud Drive share pages.

*   [SmartStrm](https://github.com/Cp0204/SmartStrm)

    A tool for generating STRM files to enable streaming from Quark Cloud Drive without downloading.

### Third-Party Open Source Projects

> [!TIP]
> These projects are developed and maintained by the community and are not directly affiliated with the QAS author.  Review risks before deploying to a production environment.
>
> If you have a project to add, please submit a pull request.

*   [nonebot-plugin-quark-autosave](https://github.com/fllesser/nonebot-plugin-quark-autosave)

    A QAS Telegram bot for managing auto-save tasks.

*   [Astrbot_plugin_quarksave](https://github.com/lm379/astrbot_plugin_quarksave)

    AstrBot plugin utilizing quark\_auto\_save for automating file transfers to Quark Cloud Drive.

## Support the Project

If this project has been useful to you, consider a small donation.

![WeChatPay](https://cdn.jsdelivr.net/gh/Cp0204/Cp0204@main/img/wechat_pay_qrcode.png)

## Disclaimer

This project is developed as a personal interest project to enhance efficiency with Quark Cloud Drive. It does not engage in any cracking activities and relies on the existing Quark API. The author is not responsible for the contents of Quark Cloud Drive or any changes to the Quark API. Use at your own discretion.

This is open-source software for educational and collaborative purposes only. It is not for commercial use and should not be used for illegal activities.

## Sponsor

CDN acceleration and security protection for this project are sponsored by Tencent EdgeOne.

<a href="https://edgeone.ai/?from=github" target="_blank"><img title="Best Asian CDN, Edge, and Secure Solutions - Tencent EdgeOne" src="https://edgeone.ai/media/34fe3a45-492d-4ea4-ae5d-ea1087ca7b4b.png" width="300"></a>
```
Key improvements and SEO optimization:

*   **Concise Hook:** The first sentence immediately grabs the reader's attention and states the core benefit.
*   **Keyword-rich headings:** Using keywords like "Quark Auto-Save," "Cloud Drive," and "Automation" improves searchability.
*   **Bulleted Features:**  Clearly lists the key benefits in a user-friendly, scannable format.
*   **Action-Oriented Language:**  Phrases like "Automate," "Manage," and "Easily" encourage engagement.
*   **Clear Installation Instructions:**  The Docker deployment section is emphasized as the recommended method.
*   **Structured Content:**  Uses headings, subheadings, and lists for better readability and SEO.
*   **Internal Links:**  Links to other parts of the documentation within the README.
*   **External Links:** Links back to the original repository and other resources, including the wiki, project documentation, and plugins
*   **Clear Warnings:** Includes important usage warnings to protect users and the Quark servers.
*   **Concise Disclaimer:** Makes the project's purpose and limitations clear.
*   **Sponsor section:** Properly displays the sponsor.