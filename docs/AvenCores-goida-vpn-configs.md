<div align="center">
    <a href="https://www.youtube.com/@avencores/" target="_blank">
      <img src="https://github.com/user-attachments/assets/338bcd74-e3c3-4700-87ab-7985058bd17e" alt="YouTube" height="40">
    </a>
    <a href="https://t.me/avencoresyt" target="_blank">
      <img src="https://github.com/user-attachments/assets/939f8beb-a49a-48cf-89b9-d610ee5c4b26" alt="Telegram" height="40">
    </a>
    <a href="https://vk.com/avencoresvk" target="_blank">
      <img src="https://github.com/user-attachments/assets/dc109dda-9045-4a06-95a5-3399f0e21dc4" alt="VK" height="40">
    </a>
    <a href="https://dzen.ru/avencores" target="_blank">
      <img src="https://github.com/user-attachments/assets/bd55f5cf-963c-4eb8-9029-7b80c8c11411" alt="Dzen" height="40">
    </a>
</div>

# Goida VPN Configs: Get Up-to-Date VPN Configurations for Bypass Restrictions

This repository ([link to original repo](https://github.com/AvenCores/goida-vpn-configs)) provides a continuously updated collection of public VPN configurations for various VPN protocols, ensuring you always have access to working servers.

---

## Key Features

*   **Automatic Updates:** Configurations are refreshed every 9 minutes via GitHub Actions.
*   **Multiple Protocols:** Supports V2Ray, VLESS, Trojan, VMess, Reality, and Shadowsocks.
*   **Wide Compatibility:** Works with popular VPN clients like v2rayNG, NekoRay, Throne, and more.
*   **Easy Import:** Configurations are provided as TXT subscriptions, easily imported into your client.
*   **QR Code Support:** QR codes available for quick setup on Android TV and other devices.

---

## Table of Contents

*   [Key Features](#key-features)
*   [Quick Start](#quick-start)
*   [How It Works](#how-it-works)
*   [Repository Structure](#repository-structure)
*   [Local Generator Run](#local-generator-run)
*   [Guides and Troubleshooting](#guides-and-troubleshooting)
*   [License](#license)
*   [Support the Author](#support-the-author)

---

## Quick Start

1.  Copy a link from the "General List of Always Up-to-Date Configurations" section below.
2.  Import the link into your preferred VPN client.
3.  Select a server with the lowest ping and connect.

---

## How It Works

*   The [`source/main.py`](source/main.py) script downloads public subscriptions from various sources.
*   The [`frequent_update.yml`](.github/workflows/frequent_update.yml) workflow runs the script every 9 minutes via cron.
*   Results are saved in the `githubmirror/` directory and pushed to this repository.
*   Each update generates a commit like: "🚀 Update config in timezone Europe/Moscow: HH:MM | DD.MM.YYYY"

---

## Repository Structure

```text
githubmirror/        — Generated .txt configs (23 files)
qr-codes/            — PNG versions of configs for QR import
source/              — Python script and generator dependencies
 ├─ main.py
 └─ requirements.txt
.github/workflows/   — CI/CD (auto-update every 9 min)
README.md            — This file
```

---

## Local Generator Run

```bash
git clone https://github.com/AvenCores/goida-vpn-configs
cd goida-vpn-configs/source
python -m pip install -r requirements.txt
export MY_TOKEN=<GITHUB_TOKEN>   # Token with repo permissions to push changes
python main.py                  # Configs will appear in ../githubmirror
```

> **Important:** In `source/main.py`, manually set `REPO_NAME = "<username>/<repository>"` if running the script from a fork.

---

## Guides and Troubleshooting

<details>
<summary>📋 General List of Always Up-to-Date Configurations</summary>
    
> Recommended lists: **[6](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/6.txt)**, **[22](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/22.txt)**, **[23](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/23.txt)**, **[24](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/24.txt)** и **[25](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/25.txt)**.

1)  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/1.txt`
2)  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/2.txt`
3)  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/3.txt`
4)  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/4.txt`
5)  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/5.txt`
6)  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/6.txt`
7)  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/7.txt`
8)  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/8.txt`
9)  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/9.txt`
10) `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/10.txt`
11) `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/11.txt`
12) `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/12.txt`
13) `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/13.txt`
14) `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/14.txt`
15) `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/15.txt`
16) `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/16.txt`
17) `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/17.txt`
18) `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/18.txt`
19) `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/19.txt`
20) `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/20.txt`
21) `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/21.txt`
22) `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/22.txt`
23) `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/23.txt`
24) `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/24.txt`
25) `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/25.txt`

🔗 [Link to QR codes of always up-to-date configs](https://github.com/AvenCores/goida-vpn-configs/tree/main/qr-codes)
</details>

<details>
<summary>📱 Guide for Android</summary>
... (content from the original README -  simplified for brevity)
</details>

<details>
<summary>📺 Guide for Android TV</summary>
... (content from the original README -  simplified for brevity)
</details>

<details>
<summary>⚠ If there is no internet connection when connecting to VPN in v2rayNG</summary>
... (content from the original README -  simplified for brevity)
</details>

<details>
<summary>⚠ If the configs did not appear when adding a VPN in v2rayNG</summary>
... (content from the original README -  simplified for brevity)
</details>

<details>
<summary>⚠ Fix for the error "Cбой проверки интернет-соединения: net/http: 12X handshake timeout"</summary>
... (content from the original README -  simplified for brevity)
</details>

<details>
<summary>⚠ Fix for the error "Fail to detect internet connection: io: read/write closed pipe"</summary>
... (content from the original README -  simplified for brevity)
</details>

<details>
<summary>🔄 Updating configs in v2rayNG</summary>
... (content from the original README -  simplified for brevity)
</details>

<details>
<summary>🖥 Guide for Windows, Linux</summary>
... (content from the original README -  simplified for brevity)
</details>

<details>
<summary>⚠ Correcting MSVCP and VCRUNTIME errors on Windows 10/11</summary>
... (content from the original README -  simplified for brevity)
</details>

<details>
<summary>🔄 Updating configs in NekoRay</summary>
... (content from the original README -  simplified for brevity)
</details>

<details>
<summary>☎ Guide for iOS, iPadOS</summary>
... (content from the original README -  simplified for brevity)
</details>

<details>
<summary>🔄 Updating configs in V2Box - V2ray Client</summary>
... (content from the original README -  simplified for brevity)
</details>

<details>
<summary>💻 Guide for MacOS</summary>
... (content from the original README -  simplified for brevity)
</details>

<details>
<summary>🔄 Updating configs in Hiddify</summary>
... (content from the original README -  simplified for brevity)
</details>

---

## License

This project is licensed under the GPL-3.0 License. See the [`LICENSE`](LICENSE) file for details.

---

## Support the Author

*   **SBER**: `2202 2050 7215 4401`