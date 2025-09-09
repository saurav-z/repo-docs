<div align="center">

![quark-logo](img/icon.png)

# Quark Auto-Save: Automate Your Quark Cloud Drive!

This powerful tool automatically transfers files, organizes filenames, and refreshes your media library, making managing your Quark Cloud Drive a breeze.  Find the original repo [here](https://github.com/Cp0204/quark-auto-save).

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
> ⛔️⛔️⛔️ **Important:** Do not set an excessively high scheduling frequency to avoid account risks and putting unnecessary pressure on Quark's servers. Every action matters.

> [!NOTE]
> The developer is not customer service; open source is free and does not mean help with using the problem; The Wiki of this project is already relatively perfect, if you encounter problems, please first read the Issues and Wiki, do not blindly ask questions.

## Key Features

*   **Automated Transfers:**
    *   Automatically save files from shared links.
    *   Supports subdirectories within shared links.
    *   Records and skips invalid shared links.
    *   Supports shared links requiring extraction codes.
    *   Intelligent resource searching and auto-filling.

*   **File Management:**
    *   Automatically creates target directories if they don't exist.
    *   Skips files that have already been transferred.
    *   Filters filenames using regular expressions for selective transfer.
    *   Organizes filenames after transfer with regex replacement.
    *   Optional file extension filtering.

*   **Task Management:**
    *   Supports multiple tasks.
    *   Sets task end dates to prevent continuous execution.
    *   Allows specific execution days for subtasks.

*   **Media Library Integration:**
    *   Searches Emby media library based on task names.
    *   Automatically refreshes the Emby library after transfers.
    *   Modular media library support for custom hook plugins.

*   **Additional Features:**
    *   Daily check-in to claim space.
    *   Supports multiple notification channels.
    *   Multi-account support (transfers from the first account only).

## Deployment

### Docker Deployment

Docker deployment provides a WebUI for management and configuration, which can meet the needs of most users.
To deploy, run the following command:

```shell
docker run -d \
  --name quark-auto-save \
  -p 5005:5005 \ # Port mapping, the number before : can be changed, i.e., the port accessed after deployment, the number after : cannot be changed
  -e WEBUI_USERNAME=admin \
  -e WEBUI_PASSWORD=admin123 \
  -v ./quark-auto-save/config:/app/config \ # Required, configuration persistence
  -v ./quark-auto-save/media:/media \ # Optional, used by module alist_strm_gen to generate strm
  --network bridge \
  --restart unless-stopped \
  cp0204/quark-auto-save:latest
  # registry.cn-shenzhen.aliyuncs.com/cp0204/quark-auto-save:latest # Domestic mirror address
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

Manage the address: http://yourhost:5005

| Environment Variable | Default      | Notes                                    |
| -------------------- | ------------ | ---------------------------------------- |
| `WEBUI_USERNAME`     | `admin`      | Management account                       |
| `WEBUI_PASSWORD`     | `admin123`   | Management password                      |
| `PORT`               | `5005`       | Management backend port                  |
| `PLUGIN_FLAGS`       |              | Plugin flags, such as `-emby,-aria2` disables some plugins |

#### One-click Update

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

| pattern                                | replace                 | Effect                                                                   |
| -------------------------------------- | ----------------------- | ---------------------------------------------------------------------- |
| `.*`                                   |                         | Transfer all files, without organizing                                               |
| `\.mp4$`                               |                         | Transfer all `.mp4` files                                             |
| `^【电影TT】花好月圆(\d+)\.(mp4\|mkv)` | `\1.\2`                 | 【电影TT】花好月圆01.mp4 → 01.mp4<br>【电影TT】花好月圆02.mkv → 02.mkv |
| `^(\d+)\.mp4`                          | `S02E\1.mp4`            | 01.mp4 → S02E01.mp4<br>02.mp4 → S02E02.mp4                             |
| `$TV`                                  |                         | [Magic match](#魔法匹配) episode files                                          |
| `^(\d+)\.mp4`                          | `{TASKNAME}.S02E\1.mp4` | 01.mp4 → 任务名.S02E01.mp4                                             |

More regex usage instructions: [Regex processing tutorial](https://github.com/Cp0204/quark-auto-save/wiki/Regex processing tutorial)

> [!TIP]
>
> **Magic Match and Magic Variables**: In the regular expression processing, we have defined some "magic match" patterns. If the value of the expression starts with `$` and the replacement is empty, the program will automatically use the preset regular expression for matching and replacement.
>
> Since v0.6.0, it supports more "magic variables" enclosed in `{}` which I call, and can be used to rename more flexibly.
>
> For more instructions, please see [Magic Match and Magic Variables](https://github.com/Cp0204/quark-auto-save/wiki/Magic Match and Magic Variables)

### Refreshing the Media Library

The feature will trigger the corresponding function when there are new transfers, such as automatically refreshing the media library, generating .strm files, etc. Configuration guide: [Plugin configuration](https://github.com/Cp0204/quark-auto-save/wiki/Plugin configuration)

The media library module is integrated as a plugin. If you are interested, please refer to [Plugin development guide](https://github.com/Cp0204/quark-auto-save/tree/main/plugins).

### More Usage Tips

Please refer to the Wiki: [Usage Tips](https://github.com/Cp0204/quark-auto-save/wiki/Usage Tips)

## Donate

If this project benefits you, you can give me 1 yuan free of charge to let me know that open source is valuable. Thank you!

![WeChatPay](https://cdn.jsdelivr.net/gh/Cp0204/Cp0204@main/img/wechat_pay_qrcode.png)

## Disclaimer

This project is developed for personal interest and aims to improve the efficiency of using the cloud disk through program automation.

The program does not have any cracking behavior, but only encapsulates the existing APIs of Quark, and all data comes from the official Quark API; the author is not responsible for the content of the cloud disk, and is not responsible for the impact caused by possible future changes of the official Quark API, please use at your own discretion.

Open source is only for learning and communication purposes, not for profit and not authorized for commercial use, and is strictly prohibited for illegal purposes.

## Sponsor

CDN acceleration and security protection for this project are sponsored by Tencent EdgeOne.

<a href="https://edgeone.ai/?from=github" target="_blank"><img title="Best Asian CDN, Edge, and Secure Solutions - Tencent EdgeOne" src="https://edgeone.ai/media/34fe3a45-492d-4ea4-ae5d-ea1087ca7b4b.png" width="300"></a>