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

# Goida VPN Configs: Get Free, Always-Updated VPN Configurations

This repository provides a constantly updated collection of public VPN configurations (V2Ray, VLESS, Trojan, VMess, Reality, Shadowsocks) to bypass internet restrictions.  [**Visit the original repo**](https://github.com/AvenCores/goida-vpn-configs).

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://github.com/AvenCores/goida-vpn-configs)
[![GPL-3.0 License](https://img.shields.io/badge/License-GPL--3.0-blue?style=for-the-badge)](./LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/AvenCores/goida-vpn-configs?style=for-the-badge)](https://github.com/AvenCores/goida-vpn-configs/issues)
[![GitHub stars](https://img.shields.io/github/stars/AvenCores/goida-vpn-configs?style=for-the-badge)](https://github.com/AvenCores/goida-vpn-configs/stargazers)
![GitHub forks](https://img.shields.io/github/forks/AvenCores/goida-vpn-configs?style=for-the-badge)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/AvenCores/goida-vpn-configs?style=for-the-badge)](https://github.com/AvenCores/goida-vpn-configs/pulls)

## Key Features:

*   **Always Up-to-Date:** Configurations are automatically updated every 9 minutes using GitHub Actions.
*   **Wide Compatibility:** Works with popular VPN clients like v2rayNG, NekoRay, Throne, v2rayN, V2Box, v2RayTun, Hiddify, and more.
*   **Easy to Use:** Simply copy and paste a link into your preferred VPN client.
*   **Multiple Protocols:** Supports V2Ray, VLESS, Trojan, VMess, Reality, and Shadowsocks protocols.
*   **QR Code Support**: QR codes available for easy configuration.

## Table of Contents

*   [Key Features](#key-features)
*   [Quick Start](#quick-start)
*   [How It Works](#how-it-works)
*   [Repository Structure](#repository-structure)
*   [Local Generator Run](#local-generator-run)
*   [Installation and Troubleshooting Video Guide](#installation-and-troubleshooting-video-guide)
*   [Guide Menu](#guide-menu)
*   [License](#license)
*   [Support the Author](#support-the-author)

## Quick Start

1.  Copy a link from the "📋 **General List of Always-Current Configurations**" section below.
2.  Import the link into your VPN client (see instructions in the Guide Menu).
3.  Select a server with the lowest ping and connect.

## How It Works

*   The script `source/main.py` downloads public subscriptions from various sources.
*   The workflow `frequent_update.yml` runs the script every 9 minutes using cron.
*   Results are saved in the `githubmirror/` directory and pushed to this repository.

Each run creates a commit like:

> 🚀 Configuration Update by Time Zone Europe/Moscow: HH:MM | DD.MM.YYYY

## Repository Structure

```text
githubmirror/        — Generated .txt configurations (25 files)
qr-codes/            — PNG versions of configs for QR import
source/              — Python script and generator dependencies
 ├─ main.py
 └─ requirements.txt
.github/workflows/   — CI/CD (auto-update every 9 min)
README.md            — This file
```

## Local Generator Run

```bash
git clone https://github.com/AvenCores/goida-vpn-configs
cd goida-vpn-configs/source
python -m pip install -r requirements.txt
export MY_TOKEN=<GITHUB_TOKEN>   # Token with repo permissions to push changes
python main.py                  # Configurations will appear in ../githubmirror
```

> **Important!**  In `source/main.py`, manually set `REPO_NAME = "<username>/<repository>"` if running the script from a fork.

## Installation and Troubleshooting Video Guide

![maxresdefault](https://github.com/user-attachments/assets/e36e2351-3b1a-4b90-87f7-cafbc74f238c)

<div align="center">

> ⚠️ **Attention!** Only the text guide below is relevant for iOS and iPadOS. The video guide is only relevant for Android, Android TV, Windows, Linux, and MacOS.

[**Watch on YouTube**](https://youtu.be/sagz2YluM70)

[**Watch on Dzen**](https://dzen.ru/video/watch/680d58f28c6d3504e953bd6d)

[**Watch on VK Video**](https://vk.com/video-200297343_456239303)

[**Watch on Telegram**](https://t.me/avencoreschat/56595)

</div>

---

## Guide Menu

<details>
  <summary>👩‍💻 Source Code for Generating Always-Current Configurations</summary>
  Link to the source code — [Link](https://github.com/AvenCores/goida-vpn-configs/tree/main/source)
</details>

---
<details>
  <summary>📋 General List of Always-Current Configurations</summary>
  > Recommended lists: **[6](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/6.txt)**, **[22](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/22.txt)**, **[23](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/23.txt)**, **[24](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/24.txt)** and **[25](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/25.txt)**.

1) `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/1.txt`
2) `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/2.txt`
3) `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/3.txt`
4) `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/4.txt`
5) `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/5.txt`
6) `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/6.txt`
7) `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/7.txt`
8) `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/8.txt`
9) `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/9.txt`
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

🔗 [Link to QR Codes of Always-Current Configurations](https://github.com/AvenCores/goida-vpn-configs/tree/main/qr-codes)
</details>

---
<details>
  <summary>📱 Guide for Android</summary>
  … (Android instructions) …
</details>

---
<details>
  <summary>📺 Guide for Android TV</summary>
  … (Android TV instructions) …
</details>

---
<details>
  <summary>⚠ If there is no internet when connected to VPN in v2rayNG</summary>
  … (Troubleshooting instructions) …
</details>

---
<details>
  <summary>⚠ If configurations do not appear when adding VPN in v2rayNG</summary>
  … (Troubleshooting instructions) …
</details>

---
<details>
  <summary>⚠ Fix for "Cбой проверки интернет-соединения: net/http: 12X handshake timeout"</summary>
  … (Troubleshooting instructions) …
</details>

---
<details>
  <summary>⚠ Fix for "Fail to detect internet connection: io: read/write closed pipe"</summary>
  … (Troubleshooting instructions) …
</details>

---
<details>
  <summary>🔄 Updating Configurations in v2rayNG</summary>
  … (Update instructions) …
</details>

---
<details>
  <summary>🖥 Guide for Windows, Linux</summary>
  … (Windows/Linux instructions) …
</details>

---
<details>
  <summary>⚠ Fixing MSVCP and VCRUNTIME Errors on Windows 10/11</summary>
  … (Troubleshooting instructions) …
</details>

---
<details>
  <summary>🔄 Updating Configurations in NekoRay</summary>
  … (Update instructions) …
</details>

---
<details>
  <summary>☎ Guide for iOS, iPadOS</summary>
  … (iOS/iPadOS instructions) …
</details>

---
<details>
  <summary>🔄 Updating Configurations in V2Box - V2ray Client</summary>
  … (Update instructions) …
</details>

---
<details>
  <summary>💻 Guide for MacOS</summary>
  … (MacOS instructions) …
</details>

---
<details>
  <summary>🔄 Updating Configurations in Hiddify</summary>
  … (Update instructions) …
</details>

---

## License

This project is licensed under the GPL-3.0 License. The full license text is in the [`LICENSE`](LICENSE) file.

---

## Support the Author

*   **SBER**: `2202 2050 7215 4401`