# Secure Your Internet with Always-Updated VPN Configurations

**Access a constantly updated collection of public VPN configurations for bypassing online restrictions with [goida-vpn-configs](https://github.com/AvenCores/goida-vpn-configs) and stay secure!**

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

## Key Features

*   **Always Fresh Configurations:** Get access to a continuously updated list of VPN configs.
*   **Multiple Protocols Supported:** Supports V2Ray, VLESS, Trojan, VMess, Reality, and Shadowsocks.
*   **Easy to Use:** Import configurations into popular VPN clients like v2rayNG, NekoRay, and more.
*   **Automated Updates:** Configs are refreshed every 9 minutes via GitHub Actions.
*   **Wide Compatibility:** Works with various devices and operating systems.

## Quick Start Guide

1.  **Choose a Config:** Select a link from the "General List of Always Up-to-Date Configurations" section below.
2.  **Import to VPN Client:** Add the selected config to your preferred VPN client (instructions below).
3.  **Connect:** Choose a server with the lowest ping and connect.

## Project Overview

This repository provides a constantly updated collection of public VPN configurations (V2Ray / VLESS / Trojan / VMess / Reality / Shadowsocks) designed to bypass online restrictions. The configs are refreshed automatically every 9 minutes using GitHub Actions, ensuring that the links in the general list are always current.

### How It Works

*   The script `source/main.py` downloads public subscriptions from various sources.
*   The workflow `.github/workflows/frequent_update.yml` runs the script every 9 minutes.
*   The results are saved in the `githubmirror/` directory and pushed to this repository.
*   Each run creates a commit with the message: "🚀 Config update by time zone Europe/Moscow: HH:MM | DD.MM.YYYY"

### Repository Structure

```text
githubmirror/        — Generated .txt configs (23 files)
qr-codes/            — PNG versions of configs for QR import
source/              — Python script and generator dependencies
 ├─ main.py
 └─ requirements.txt
.github/workflows/   — CI/CD (auto-update every 9 minutes)
README.md            — This file
```

## Local Generator Setup

```bash
git clone https://github.com/AvenCores/goida-vpn-configs
cd goida-vpn-configs/source
python -m pip install -r requirements.txt
export MY_TOKEN=<GITHUB_TOKEN>   # Token with repo permissions to push changes
python main.py                  # Configs will appear in ../githubmirror
```

>   **Important:** If running the script from a fork, manually set `REPO_NAME = "<username>/<repository>"` in `source/main.py`.

## Video Guides & Tutorials

![maxresdefault](https://github.com/user-attachments/assets/e36e2351-3b1a-4b90-87f7-cafbc74f238c)

<div align="center">

>   ⚠️ **Attention!**  Only the text guide is applicable for iOS and iPadOS. The video guide is only applicable for Android, Android TV, Windows, Linux, and MacOS.

[**Watch on YouTube**](https://youtu.be/sagz2YluM70)

[**Watch on Dzen**](https://dzen.ru/video/watch/680d58f28c6d3504e953bd6d)

[**Watch on VK Video**](https://vk.com/video-200297343_456239303)

[**Watch on Telegram**](https://t.me/avencoreschat/56595)

</div>

## General Repository Guide Menu

<details>
<summary>💻 Source Code</summary>
  The source code for generating the ever-current configurations is available at [https://github.com/AvenCores/goida-vpn-configs/tree/main/source](https://github.com/AvenCores/goida-vpn-configs/tree/main/source).
</details>

<details>
<summary>📋 General List of Always Up-to-Date Configurations</summary>

>   Recommended lists: **[6](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/6.txt)**, **[22](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/22.txt)**, **[23](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/23.txt)**, **[24](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/24.txt)**, and **[25](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/25.txt)**.

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

🔗  [Link to QR codes of always current configurations](https://github.com/AvenCores/goida-vpn-configs/tree/main/qr-codes)

</details>

<details>
<summary>📱 Android Guide</summary>
  ... (Android Instructions) ...
</details>

<details>
<summary>📺 Android TV Guide</summary>
  ... (Android TV Instructions) ...
</details>

<details>
<summary>⚠ If no internet when connecting to VPN in v2rayNG</summary>
  ... (Troubleshooting Instructions) ...
</details>

<details>
<summary>⚠ If configs don't appear when adding a VPN in v2rayNG</summary>
  ... (Troubleshooting Instructions) ...
</details>

<details>
<summary>⚠ Fix for "Connection check failed: net/http: 12X handshake timeout"</summary>
  ... (Troubleshooting Instructions) ...
</details>

<details>
<summary>⚠ Fix for "Fail to detect internet connection: io: read/write closed pipe"</summary>
  ... (Troubleshooting Instructions) ...
</details>

<details>
<summary>🔄 Updating Configurations in v2rayNG</summary>
  ... (Update Instructions) ...
</details>

<details>
<summary>🖥 Windows, Linux Guide</summary>
  ... (Windows and Linux Instructions) ...
</details>

<details>
<summary>⚠ Fixing MSVCP and VCRUNTIME errors on Windows 10/11</summary>
  ... (Troubleshooting Instructions) ...
</details>

<details>
<summary>🔄 Updating configurations in NekoRay</summary>
  ... (Update Instructions) ...
</details>

<details>
<summary>📱 iOS, iPadOS Guide</summary>
  ... (iOS and iPadOS Instructions) ...
</details>

<details>
<summary>🔄 Updating configurations in V2Box - V2ray Client</summary>
  ... (Update Instructions) ...
</details>

<details>
<summary>💻 MacOS Guide</summary>
  ... (MacOS Instructions) ...
</details>

<details>
<summary>🔄 Updating configurations in Hiddify</summary>
  ... (Update Instructions) ...
</details>

---

## License

This project is licensed under the GPL-3.0 License. See [`LICENSE`](LICENSE) for the full license text.

---

## Support the Author

*   **SBER:** `2202 2050 7215 4401`