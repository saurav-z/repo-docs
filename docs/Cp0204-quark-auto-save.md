<div align="center">

![quark-logo](img/icon.png)

# Quark Auto-Save: Automate Your Quark Drive Experience

This tool automates Quark cloud drive tasks, saving you time and effort.  Check out the [original repo](https://github.com/Cp0204/quark-auto-save) for more details.

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

## Key Features

*   **Automated Transfers:** Automatically saves and organizes files from shared links.
*   **WebUI Management:** Configure and manage your tasks easily via a web interface.
*   **Flexible File Management:**
    *   Create directories automatically.
    *   Skip already saved files.
    *   Filter files by name using regular expressions.
    *   Rename files after transfer using regular expressions.
    *   Optionally ignore file extensions.
*   **Task Management:**
    *   Support for multiple tasks.
    *   Set expiration dates for tasks.
    *   Schedule tasks to run on specific days of the week.
*   **Media Library Integration:**
    *   Integrate with Emby to refresh your media library automatically.
    *   Modular plugin system for custom media library integration.
*   **Additional Features:**
    *   Daily sign-in to claim free storage space.
    *   Multiple notification channels supported.
    *   Supports multiple accounts (saves files from the first account).

## Deployment

### Docker Deployment

Docker deployment provides a WebUI for management and configuration. Deploy using the following command:

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

Manage the application at: `http://yourhost:5005`

| Environment Variable | Default    | Notes                                         |
| -------------------- | ---------- | --------------------------------------------- |
| `WEBUI_USERNAME`     | `admin`    | Admin username                                |
| `WEBUI_PASSWORD`     | `admin123` | Admin password                                |
| `PORT`               | `5005`     | WebUI port                                    |
| `PLUGIN_FLAGS`       |            | Plugin flags, e.g., `-emby,-aria2` to disable plugins |

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

| Pattern                                | Replace                 | Effect                                                                        |
| -------------------------------------- | ----------------------- | ----------------------------------------------------------------------------- |
| `.*`                                   |                         | Transfer all files without any changes.                                       |
| `\.mp4$`                               |                         | Transfer all files with the `.mp4` extension.                                  |
| `^【电影TT】花好月圆(\d+)\.(mp4\|mkv)` | `\1.\2`                 | 【电影TT】花好月圆01.mp4 → 01.mp4<br>【电影TT】花好月圆02.mkv → 02.mkv       |
| `^(\d+)\.mp4`                          | `S02E\1.mp4`            | 01.mp4 → S02E01.mp4<br>02.mp4 → S02E02.mp4                                   |
| `$TV`                                  |                         | [Magic Matching](#magic-matching) for TV series files.                           |
| `^(\d+)\.mp4`                          | `{TASKNAME}.S02E\1.mp4` | 01.mp4 → TaskName.S02E01.mp4                                                  |

More regular expression usage: [Regex Processing Tutorial](https://github.com/Cp0204/quark-auto-save/wiki/%E6%AD%A3%E5%88%99%E5%A4%84%E7%90%86%E6%95%99%E7%A8%8B)

> [!TIP]
>
> **Magic Matching and Magic Variables**: In regex processing, we define some "magic matching" patterns, if the value of the Expression starts with `$` and the Replacement is left empty, the program will automatically use the preset regular expressions for matching and replacing.
>
> Since v0.6.0, support more "magic variables" enclosed in {}, which can be used for more flexible renaming.
>
> For more information, please see [Magic Matching and Magic Variables](https://github.com/Cp0204/quark-auto-save/wiki/%E9%AD%94%E6%B3%95%E5%8C%B9%E9%85%8D%E5%92%8C%E9%AD%94%E6%B3%95%E5%8F%98%E9%87%8F)

### Refreshing Your Media Library

When new files are saved, you can trigger actions such as automatically refreshing your media library or generating `.strm` files. Configuration guide: [Plugin Configuration](https://github.com/Cp0204/quark-auto-save/wiki/%E6%8F%92%E4%BB%B6%E9%85%8D%E7%BD%AE)

Media library modules are integrated as plugins. If you're interested, please refer to the [plugin development guide](https://github.com/Cp0204/quark-auto-save/tree/main/plugins).

### Additional Tips

Refer to the Wiki: [Usage Tips](https://github.com/Cp0204/quark-auto-save/wiki/%E4%BD%BF%E7%94%A8%E6%8A%80%E5%B7%A7%E9%9B%86%E9%94%A6)

## Donate

If you find this project useful, you can donate 1 yuan to me. Thanks!

![WeChatPay](https://cdn.jsdelivr.net/gh/Cp0204/Cp0204@main/img/wechat_pay_qrcode.png)

## Disclaimer

This project is developed for personal interest, designed to improve the efficiency of using cloud drives through automation.

The program does not involve any cracking behavior, it only encapsulates the existing APIs of Quark. All data comes from the official Quark API. The author is not responsible for the content of the cloud drive or the impact of possible future changes in the official Quark API. Please use it at your own discretion.

This project is open source for learning and exchange purposes only, and is not for profit or authorized for commercial use. It is strictly prohibited to use it for illegal purposes.

## Sponsor

CDN acceleration and security protection for this project are sponsored by Tencent EdgeOne.

<a href="https://edgeone.ai/?from=github" target="_blank"><img title="Best Asian CDN, Edge, and Secure Solutions - Tencent EdgeOne" src="https://edgeone.ai/media/34fe3a45-492d-4ea4-ae5d-ea1087ca7b4b.png" width="300"></a>