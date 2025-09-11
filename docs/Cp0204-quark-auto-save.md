<div align="center">

![quark-logo](img/icon.png)

# Quark Auto Save: Automate Your Quark Network Drive

**Automatically save, organize, and manage files in your Quark Network Drive with scheduled tasks and media library integration.**

[<img src="https://img.shields.io/github/v/release/Cp0204/quark-auto-save?logo=github" alt="GitHub release">](https://github.com/Cp0204/quark-auto-save)
[<img src="https://img.shields.io/docker/pulls/cp0204/quark-auto-save?logo=docker&&logoColor=white" alt="Docker Pulls">](https://hub.docker.com/r/cp0204/quark-auto-save)
[<img src="https://img.shields.io/docker/image-size/cp0204/quark-auto-save?logo=docker&&logoColor=white" alt="Docker Image Size">](https://hub.docker.com/r/cp0204/quark-auto-save)
[<img src="https://img.shields.io/badge/wiki-Documents-green?logo=github" alt="Wiki">](https://github.com/Cp0204/quark-auto-save/wiki)

</div>

> [!CAUTION]
> ⚠️ **Important:** Avoid excessively frequent scheduled tasks to prevent account risks and excessive server load.

> [!NOTE]
> The developer is not a customer service representative. This project's Wiki is comprehensive. Please consult the Issues and Wiki before asking questions.

## Key Features

*   **Automated Saving & Organization:** Simplify your Quark Network Drive management by automating file saving, naming, and organization.
*   **Docker Deployment with WebUI:** Deploy easily using Docker with a user-friendly WebUI for configuration.
*   **Share Link Support:** Supports saving files from share links, including those requiring extraction codes and subdirectories.
*   **Smart Resource Search:** Automatically finds and saves resources, simplifying the process of adding content to your drive.
*   **File Management:** Offers advanced file management options, including skipping existing files, filtering by name, and renaming with regular expressions.
*   **Scheduled Tasks:** Supports multiple tasks with flexible scheduling, including end dates and day-of-the-week execution.
*   **Media Library Integration:** Automatically refreshes your media library (e.g., Emby) after saving files, ensuring your content is always up-to-date.
*   **Additional Features:** Includes daily sign-in for extra space, support for various notification channels, and multi-account support.

## Getting Started

### Docker Deployment

Easily deploy and configure Quark Auto Save with Docker and manage it through the WebUI.

**Run the following command:**

```shell
docker run -d \
  --name quark-auto-save \
  -p 5005:5005 \ # 映射端口，:前的可以改，即部署后访问的端口，:后的不可改
  -e WEBUI_USERNAME=admin \
  -e WEBUI_PASSWORD=admin123 \
  -v ./quark-auto-save/config:/app/config \ # 必须，配置持久化
  -v ./quark-auto-save/media:/media \ # 可选，模块alist_strm_gen生成strm使用
  --network bridge \
  --restart unless-stopped \
  cp0204/quark-auto-save:latest
  # registry.cn-shenzhen.aliyuncs.com/cp0204/quark-auto-save:latest # 国内镜像地址
```

**or use `docker-compose.yml`**

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

**Access the WebUI at:** `http://yourhost:5005`

| Environment Variable | Default   | Description                  |
| -------------------- | --------- | ---------------------------- |
| `WEBUI_USERNAME`     | `admin`   | WebUI Username               |
| `WEBUI_PASSWORD`     | `admin123` | WebUI Password               |
| `PORT`               | `5005`    | WebUI Port                   |
| `PLUGIN_FLAGS`       |           | Disable plugins, e.g., `-emby` |

#### Update

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

| Pattern                                  | Replace                 | Result                                                                   |
| ---------------------------------------- | ----------------------- | ---------------------------------------------------------------------- |
| `.*`                                     |                         | Save all files without renaming.                                         |
| `\.mp4$`                                 |                         | Save all files with the `.mp4` extension.                               |
| `^【电影TT】花好月圆(\d+)\.(mp4\|mkv)`     | `\1.\2`                 | 【电影TT】花好月圆01.mp4 → 01.mp4<br>【电影TT】花好月圆02.mkv → 02.mkv |
| `^(\d+)\.mp4`                            | `S02E\1.mp4`            | 01.mp4 → S02E01.mp4<br>02.mp4 → S02E02.mp4                             |
| `$TV`                                    |                         | [Magic Matching](#魔法匹配) of episode files.                                    |
| `^(\d+)\.mp4`                            | `{TASKNAME}.S02E\1.mp4` | 01.mp4 → 任务名.S02E01.mp4                                             |

For more information, refer to the [Regular Expression Tutorial](https://github.com/Cp0204/quark-auto-save/wiki/正则处理教程).

> [!TIP]
> **Magic Matching and Magic Variables:** Define "magic matching" patterns using regex in rename rules. If a "Replace" is empty and the "Expression" starts with `$`, the system uses a pre-configured regex for matching and replacing.
>
> From v0.6.0, more "magic variables" using `{}` are supported for more flexible renaming.
>
> Read more at [Magic Matching and Magic Variables](https://github.com/Cp0204/quark-auto-save/wiki/魔法匹配和魔法变量).

### Media Library Integration

Integrate with media libraries like Emby to automatically refresh them after saving files. See the [Plugin Configuration](https://github.com/Cp0204/quark-auto-save/wiki/插件配置) guide.

For plugin development, see the [Plugin Development Guide](https://github.com/Cp0204/quark-auto-save/tree/main/plugins).

### More Tips and Tricks

Find more tips in the [Tips and Tricks](https://github.com/Cp0204/quark-auto-save/wiki/使用技巧集锦) section of the Wiki.

## Support the Project

If this project benefits you, consider a small donation.  Thank you!

![WeChatPay](https://cdn.jsdelivr.net/gh/Cp0204/Cp0204@main/img/wechat_pay_qrcode.png)

## Disclaimer

This project is developed for personal use and to improve Quark Network Drive efficiency through automation.

It does not involve any cracking behaviors. The program encapsulates Quark's existing APIs, with all data originating from Quark's official APIs. I am not responsible for the content on the drive or the potential impact of future changes to the Quark API. Please use this project responsibly.

Open-source for learning and communication only. It is not for commercial use or profit, and is strictly forbidden for illegal purposes.

## Sponsor

CDN acceleration and security protection for this project are sponsored by Tencent EdgeOne.

<a href="https://edgeone.ai/?from=github" target="_blank"><img title="Best Asian CDN, Edge, and Secure Solutions - Tencent EdgeOne" src="https://edgeone.ai/media/34fe3a45-492d-4ea4-ae5d-ea1087ca7b4b.png" width="300"></a>

---

**For the original repository, visit:** [Cp0204/quark-auto-save](https://github.com/Cp0204/quark-auto-save)
```
Key improvements and explanations:

*   **SEO Optimization:** Added keywords like "Quark Network Drive," "automation," "save," "organize," "Docker," and "media library" in headings and descriptions to improve search engine visibility.  Also added the term "Auto Save" to the name.
*   **Concise Hook:** Replaced the initial description with a more compelling and concise one-sentence hook to immediately grab the user's attention.
*   **Structured Headings:** Used clear and consistent headings (like "Key Features," "Getting Started," and "Usage") to improve readability and organization.
*   **Bulleted Key Features:** Highlighted key features using bullet points, making the information easy to scan and understand.
*   **Clear Instructions:** Provided a more straightforward "Getting Started" section, with the Docker command formatted more clearly and instructions on how to access the WebUI.
*   **Emphasis on Best Practices:** Re-emphasized warnings about not running tasks too frequently.
*   **Wiki Links:** Kept all the links to the wiki and other important resources, and added some extra ones.
*   **Call to Action (Support):** Kept the option for donations.
*   **Disclaimer:** Preserved and enhanced the disclaimer for legal and ethical considerations.
*   **Sponsor:** Kept the sponsor section.
*   **Backlink:** Added a clear backlink to the original repository at the end.
*   **Formatting:** Improved overall Markdown formatting for better readability.
*   **Replaced old Chinese with English.**
*   **Reordered the sections for better flow.**
*   **Expanded on the descriptions for clarity.**