<div align="center">
  <img src="img/icon.png" alt="Quark Auto-Save Logo" width="100">

  # Quark Auto-Save: Automate Your Quark Network Disk Transfers

  **Automatically transfer, organize, and refresh your Quark Network Disk content with ease, saving you time and keeping your media library up-to-date.**  This project streamlines the process of managing your Quark Network Disk content.
</div>

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

![run_log](img/run_log.png)

> [!CAUTION]
> ⛔️⛔️⛔️ **Warning:** Resources are not updated constantly. **Avoid setting excessively frequent scheduled runs!** This can lead to account risks and put unnecessary strain on Quark's servers.

> [!NOTE]
> The developer is not customer support. Open source and free does not mean help with usage problems; The project wiki is relatively complete, encounter problems first, read Issues and Wiki , please do not ask blindly.

## Key Features

*   **Automated Transfers:** Automatically saves files from Quark Network Disk shares.
*   **Share Link Support:** Supports subdirectories within share links and handles links requiring extraction codes.
*   **Smart File Handling:** Skips already saved files, filters filenames, and organizes file names using regular expressions.
*   **Media Library Integration:** Integrates with media servers like Emby for automatic library refreshes and series tracking.
*   **Task Management:** Supports multiple tasks with individual scheduling options.
*   **Daily Sign-in & Notifications:**  Includes daily sign-in for space and supports multiple notification channels.
*   **Docker Deployment:**  Easy deployment with WebUI configuration for simplified setup.

## Deployment

### Docker Deployment

Docker deployment provides WebUI management configuration, and graphical configuration can meet most of the needs. Deployment command:

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

Management Address: http://yourhost:5005

| Environment Variable | Default      | Notes                                     |
| -------------------- | ------------ | ----------------------------------------- |
| `WEBUI_USERNAME`     | `admin`      | Management account                        |
| `WEBUI_PASSWORD`     | `admin123`   | Management password                       |
| `PORT`               | `5005`       | Management background port                |
| `PLUGIN_FLAGS`       |              | Plugin flag, such as `-emby,-aria2` disable some plugins |

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

| Pattern                                | Replace                 | Effect                                                                   |
| -------------------------------------- | ----------------------- | ---------------------------------------------------------------------- |
| `.*`                                   |                         | Transfer all files without processing                                     |
| `\.mp4$`                               |                         | Transfer all `.mp4` files                                               |
| `^【电影TT】花好月圆(\d+)\.(mp4\|mkv)` | `\1.\2`                 | 【电影TT】花好月圆01.mp4 → 01.mp4<br>【电影TT】花好月圆02.mkv → 02.mkv |
| `^(\d+)\.mp4`                          | `S02E\1.mp4`            | 01.mp4 → S02E01.mp4<br>02.mp4 → S02E02.mp4                             |
| `$TV`                                  |                         | [Magic Matching](#magic-matching) episode files                                          |
| `^(\d+)\.mp4`                          | `{TASKNAME}.S02E\1.mp4` | 01.mp4 → 任务名.S02E01.mp4                                             |

More regex usage instructions: [Regex processing tutorial](https://github.com/Cp0204/quark-auto-save/wiki/Regex processing tutorial)

> [!TIP]
>
> **Magic Matching and Magic Variables**: In regular expression processing, we have defined some "magic matching" patterns. If the value of the expression starts with $ and the replace expression is left blank, the program will automatically use the preset regular expression for matching and replacing.
>
> Since v0.6.0, more "magic variables" wrapped in {} are supported, which can be used to rename more flexibly.
>
> For more instructions, see [Magic Matching and Magic Variables](https://github.com/Cp0204/quark-auto-save/wiki/magic-matching-and-magic-variables)

### Refreshing the Media Library

When new transfers are made, the corresponding functions can be triggered, such as automatically refreshing the media library, generating .strm files, etc. Configuration guide: [Plugin Configuration](https://github.com/Cp0204/quark-auto-save/wiki/plugin configuration)

The media library module is integrated in the form of a plugin. If you are interested, please refer to the [plugin development guide](https://github.com/Cp0204/quark-auto-save/tree/main/plugins).

### More Usage Tips

Please refer to the Wiki: [Tips for Usage](https://github.com/Cp0204/quark-auto-save/wiki/Tips-for-Usage)

## Ecosystem Projects

The following shows the QAS ecosystem projects, including official projects and third-party projects.

### Official Projects

* [QAS One-Click Push Assistant](https://greasyfork.org/zh-CN/scripts/533201-qas一键推送助手)

  Greasy Fork script, add a button to push to QAS on the Quark Network Disk sharing page

* [SmartStrm](https://github.com/Cp0204/SmartStrm)

  STRM file generator, used for post-transfer processing, media-free download into the library playback.

### Third-Party Open Source Projects

> [!TIP]
>
> The following third-party open source projects are developed and maintained by the community and are not directly affiliated with the QAS author. Before deploying to a production environment, please evaluate the relevant risks yourself.
>
> If you have a new project that is not listed here, you can submit it via Issues.

* [nonebot-plugin-quark-autosave](https://github.com/fllesser/nonebot-plugin-quark-autosave)

  QAS Telegram robot, quickly manage automatic transfer tasks

* [Astrbot_plugin_quarksave](https://github.com/lm379/astrbot_plugin_quarksave)

  AstrBot plugin, call quark_auto_save to automatically transfer resources to Quark Network Disk

## Donations

If you benefit from this project, you can give me 1 yuan for free, let me know that open source is valuable. Thank you!

![WeChatPay](https://cdn.jsdelivr.net/gh/Cp0204/Cp0204@main/img/wechat_pay_qrcode.png)

## Disclaimer

This project is for personal interest development, aimed at improving the efficiency of network disk usage through program automation.

The program does not have any cracking behavior, it is only a package of the existing APIs of Quark, all data comes from Quark's official API; I am not responsible for the content of the network disk, nor am I responsible for the impact caused by the future possible changes of the Quark official API, please use it at your own discretion.

Open source is only for learning and communication use, not for profit and not authorized for commercial use, strictly prohibited for illegal purposes.

## Sponsor

CDN acceleration and security protection for this project are sponsored by Tencent EdgeOne.

<a href="https://edgeone.ai/?from=github" target="_blank"><img title="Best Asian CDN, Edge, and Secure Solutions - Tencent EdgeOne" src="https://edgeone.ai/media/34fe3a45-492d-4ea4-ae5d-ea1087ca7b4b.png" width="300"></a>
```
Key improvements and SEO considerations:

*   **Concise Hook:** A clear and inviting one-sentence hook to grab the reader's attention.
*   **Clear Headings:** Uses H1, H2, and H3 headings to organize the information and improve readability for both users and search engines.
*   **Bulleted Feature List:** Highlights key features for easy scanning and comprehension.
*   **SEO Keywords:**  Includes relevant keywords like "Quark Network Disk," "automation," "transfer," "media library," and "Docker."
*   **Contextual Links:**  Links back to the original GitHub repository for easy access to the project.
*   **Concise & Focused Language:** Streamlined the text, removed redundancies, and focused on the core benefits.
*   **Stronger Call to Action:** Directly indicates the benefit to the user.
*   **Docker Emphasis:** Improved the Docker deployment section and added a Docker Compose example to aid in easier setup.
*   **Clear Instructions:** Better formatting and clarity in the usage sections.
*   **Simplified Ecosystem:** Highlights the key ecosystem components.
*   **More Specific Feature Breakdown**: Broke down some features to provide more detail for the user.

This improved README provides a much better overview of the project and is more likely to attract and inform users, as well as be more discoverable in search results.