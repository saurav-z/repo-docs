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

# Goida VPN Configs: Get Instant Access to Up-to-Date VPN Configurations

Tired of online restrictions?  This repository provides a regularly updated collection of public VPN configurations for `V2Ray`, `VLESS`, `Trojan`, `VMess`, `Reality`, and `Shadowsocks`, ready for immediate use!  ([Original Repo](https://github.com/AvenCores/goida-vpn-configs))

## Key Features:

*   **Always Up-to-Date:** Configurations are automatically updated every 9 minutes using GitHub Actions, ensuring you have the latest working options.
*   **Wide Compatibility:**  Works with most modern VPN clients, including `v2rayNG`, `NekoRay`, `Throne`, and `V2Box`.
*   **Multiple Protocols:** Supports a range of VPN protocols for flexibility and bypassing various restrictions.
*   **Easy to Use:** Simply copy and paste a config link into your VPN client.
*   **QR Code Support:**  QR codes are provided for easy import on Android TV and other devices.

## Quick Start:

1.  Choose a config link from the "General List of Always Up-to-Date Configurations" section.
2.  Import the link into your preferred VPN client.
3.  Connect to a server with the lowest ping for optimal performance.

## Contents:

*   [Quick Start](#quick-start)
*   [How it Works](#how-it-works)
*   [Repository Structure](#repository-structure)
*   [Local Generator Run](#local-generator-run)
*   [Video Guide](#-видео-гайд-по-установке-и-решению-проблем)
*   [Guide Menu](#-общее-меню-гайдов-репозитория)
*   [License](#-лицензия)
*   [Support the Author](#-поддержать-автора)

---

## How it Works:

*   The [`source/main.py`](source/main.py) script downloads public subscriptions from multiple sources.
*   The [`frequent_update.yml`](.github/workflows/frequent_update.yml) workflow runs the script every 9 minutes.
*   Results are saved to the `githubmirror/` directory and automatically pushed to this repository.

Each update creates a commit with a message like:
> 🚀 Обновление конфига по часовому поясу Европа/Москва: HH:MM | DD.MM.YYYY

## Repository Structure:

```
githubmirror/        — Generated .txt configs (23 files)
qr-codes/            — PNG versions of configs for QR import
source/              — Python script and generator dependencies
 ├─ main.py
 └─ requirements.txt
.github/workflows/   — CI/CD (auto-update every 9 minutes)
README.md            — This file
```

## Local Generator Run:

1.  `git clone https://github.com/AvenCores/goida-vpn-configs`
2.  `cd goida-vpn-configs/source`
3.  `python -m pip install -r requirements.txt`
4.  `export MY_TOKEN=<GITHUB_TOKEN>` (requires a GitHub token with repo permissions)
5.  `python main.py` (Configs will appear in the ../githubmirror directory)

> **Important:** If running the script from a fork, manually set `REPO_NAME = "<username>/<repository>"` in `source/main.py`.

---

# 🎦 Video Guide

![maxresdefault](https://github.com/user-attachments/assets/e36e2351-3b1a-4b90-87f7-cafbc74f238c)

<div align="center">

> ⚠️ **Attention!** The text guide below is only relevant for iOS and iPadOS.  The video guide is applicable for Android, Android TV, Windows, Linux, and MacOS.

[**Watch on YouTube**](https://youtu.be/sagz2YluM70)

[**Watch on Dzen**](https://dzen.ru/video/watch/680d58f28c6d3504e953bd6d)

[**Watch on VK Video**](https://vk.com/video-200297343_456239303)

[**Watch on Telegram**](https://t.me/avencoreschat/56595)

</div>

---

# 🗂️ Guide Menu

<details>

<summary>👩‍💻 Source Code for Generating Always Up-to-Date Configurations</summary>

Link to Source Code — [Link](https://github.com/AvenCores/goida-vpn-configs/tree/main/source)

</details>

---
<details>

<summary>📋 General List of Always Up-to-Date Configurations</summary>

> Recommended Lists: **[6](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/6.txt)**, **[22](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/22.txt)**, **[23](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/23.txt)**, **[24](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/24.txt)** and **[25](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/25.txt)**.

1.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/1.txt`
2.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/2.txt`
3.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/3.txt`
4.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/4.txt`
5.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/5.txt`
6.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/6.txt`
7.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/7.txt`
8.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/8.txt`
9.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/9.txt`
10. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/10.txt`
11. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/11.txt`
12. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/12.txt`
13. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/13.txt`
14. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/14.txt`
15. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/15.txt`
16. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/16.txt`
17. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/17.txt`
18. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/18.txt`
19. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/19.txt`
20. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/20.txt`
21. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/21.txt`
22. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/22.txt`
23. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/23.txt`
24. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/24.txt`
25. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/25.txt`

🔗 [Link to QR Codes of Always Up-to-Date Configurations](https://github.com/AvenCores/goida-vpn-configs/tree/main/qr-codes)
</details>

---
<details>

<summary>📱 Guide for Android</summary>

**1.** Download **"v2rayNG"** — [Link](https://github.com/2dust/v2rayNG/releases/download/1.10.19/v2rayNG_1.10.19_universal.apk)

**2.** Copy to clipboard:

-   [ ] **Always up-to-date**

    > Recommended Lists: **[6](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/6.txt)**, **[22](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/22.txt)**, **[23](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/23.txt)**, **[24](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/24.txt)** and **[25](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/25.txt)**.

    1.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/1.txt`
    2.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/2.txt`
    3.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/3.txt`
    4.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/4.txt`
    5.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/5.txt`
    6.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/6.txt`
    7.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/7.txt`
    8.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/8.txt`
    9.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/9.txt`
    10. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/10.txt`
    11. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/11.txt`
    12. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/12.txt`
    13. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/13.txt`
    14. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/14.txt`
    15. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/15.txt`
    16. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/16.txt`
    17. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/17.txt`
    18. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/18.txt`
    19. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/19.txt`
    20. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/20.txt`
    21. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/21.txt`
    22. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/22.txt`
    23. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/23.txt`
    24. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/24.txt`
    25. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/25.txt`

**3.** Go to the **"v2rayNG"** app and tap the ➕ in the top right corner, then select **"Import from clipboard"**.

**4.** Tap **"three dots in the top right"**, then **"Group profile check"**, and after the check is complete, tap **"Sort by test results"** in the same menu.

**5.** Select the server you need and tap the ▶️ button in the bottom right corner.

</details>

<details>

<summary>📺 Guide for Android TV</summary>

**1.** Download **"v2rayNG"** — [Link](https://github.com/2dust/v2rayNG/releases/download/1.10.19/v2rayNG_1.10.19_universal.apk)

> Recommended **"QR codes"**: **[6](https://github.com/AvenCores/goida-vpn-configs/blob/main/qr-codes/6.png)**, **[22](https://github.com/AvenCores/goida-vpn-configs/blob/main/qr-codes/22.png)**, **[23](https://github.com/AvenCores/goida-vpn-configs/blob/main/qr-codes/23.png)**, **[24](https://github.com/AvenCores/goida-vpn-configs/blob/main/qr-codes/24.png)** and **[25](https://github.com/AvenCores/goida-vpn-configs/blob/main/qr-codes/25.png)**.

**2.** Download the **"QR codes"** of the always up-to-date configurations — [Link](https://github.com/AvenCores/goida-vpn-configs/tree/main/qr-codes)

**3**. Go to the **"v2rayNG"** app and tap the ➕ in the top right corner, then select **"Import from QR code"**, select the image by tapping the photo icon in the top right corner.

**4.** Tap **"three dots in the top right"**, then **"Group profile check"**, and after the check is complete, tap **"Sort by test results"** in the same menu.

**5.** Select the server you need and tap the ▶️ button in the bottom right corner.

</details>

<details>

<summary>⚠ If there is no internet when connecting to VPN in v2rayNG</summary>

Link to a video demonstrating the fix — [Link](https://t.me/avencoreschat/25254)

</details>

<details>

<summary>⚠ If the configurations did not appear when adding VPN to v2rayNG</summary>

**1.** Tap the **"three bars"** in the **"top left corner"**.

**2.** Tap the **"Groups"** button.

**3.** Tap the **"circle icon with an arrow"** in the **"top right corner"** and wait for the update to finish.

</details>

<details>

<summary>⚠ Fix error "Failed to check internet connection: net/http: 12X handshake timeout"</summary>

**1.** On the desktop, hold down the icon of **"v2rayNG"** and tap the **"About app"** item.

**2.** Tap the **"Stop"** button and restart **"v2rayNG"**.

</details>

<details>

<summary>⚠ Fix error "Fail to detect internet connection: io: read/write closed pipe"</summary>

**1.** On the desktop, hold down the icon of **"v2rayNG"** and tap the **"About app"** item.

**2.** Tap the **"Stop"** button and restart **"v2rayNG"**.

**3.** Tap **"three dots in the top right"**, then **"Group profile check"**, and after the check is complete, tap **"Sort by test results"** in the same menu.

**4.** Select the server you need and tap the ▶️ button in the bottom right corner.

</details>

<details>

<summary>🔄 Updating configurations in v2rayNG</summary>

**1.** Tap the **"three bars icon"** in the **"top left corner"**.

**2.** Select the **"Groups"** tab.

**3.** Tap the **"circle icon with an arrow"** in the **"top right corner"**.

</details>

---
<details>

<summary>🖥 Guide for Windows, Linux</summary>

**1.** Download **"Throne"** — [Windows 10/11](https://github.com/throneproj/Throne/releases/download/1.0.5/Throne-1.0.5-windows64.zip) / [Windows 7/8/8.1](https://github.com/throneproj/Throne/releases/download/1.0.5/Throne-1.0.5-windowslegacy64.zip) / [Linux](https://github.com/throneproj/Throne/releases/download/1.0.5/Throne-1.0.5-linux-amd64.zip)

**2.** Copy to clipboard:

-   [ ] **Always up-to-date**

    > Recommended Lists: **[6](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/6.txt)**, **[22](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/22.txt)**, **[23](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/23.txt)**, **[24](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/24.txt)** and **[25](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/25.txt)**.

    1.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/1.txt`
    2.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/2.txt`
    3.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/3.txt`
    4.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/4.txt`
    5.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/5.txt`
    6.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/6.txt`
    7.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/7.txt`
    8.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/8.txt`
    9.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/9.txt`
    10. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/10.txt`
    11. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/11.txt`
    12. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/12.txt`
    13. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/13.txt`
    14. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/14.txt`
    15. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/15.txt`
    16. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/16.txt`
    17. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/17.txt`
    18. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/18.txt`
    19. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/19.txt`
    20. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/20.txt`
    21. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/21.txt`
    22. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/22.txt`
    23. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/23.txt`
    24. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/24.txt`
    25. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/25.txt`

**3.** Tap **"Profiles"**, and then **"Add profile from clipboard"**.

**4.** Select all configs with the key combination **"Ctrl + A"**, tap **"Profiles"** in the top menu, and then **"Latency Test (ping) of the selected profile"** and wait for the test to finish (the inscription **"Latency Test (ping) is complete!"** will appear in the **"Logs"** tab)

**5.** Click on the **"Latency (ping)"** column.

**6.** In the upper part of the program, activate the **"TUN mode"** option by checking the box.

**7.** Select one of the configurations with the lowest **"Latency (ping)"**, and then tap **"LMB"** and **"Run"**.

</details>

<details>

<summary>⚠ Fixing the MSVCP and VCRUNTIME error on Windows 10/11</summary>

**1.** Press **"Win+R"** and type **"control"**.

**2.** Select **"Programs and features"**.

**3.** In the search (top right) type the word **"Visual"** and delete everything related to **"Microsoft Visual"**.

**4.** Download the archive and unpack it — [Link](https://cf.comss.org/download/Visual-C-Runtimes-All-in-One-Jul-2025.zip)

**5.** Run **"install_bat.all"** from *Administrator* and wait for everything to install.

</details>

<details>

<summary>🔄 Updating configurations in NekoRay</summary>

**1.** Click on the **"Settings"** button.

**2.** Select **"Groups"**.

**3.** Click on the **"Update all subscriptions"** button.

</details>

---
<details>

<summary>📱 Guide for iOS, iPadOS</summary>

**1.** Download **"V2Box - V2ray Client"** — [Link](https://apps.apple.com/ru/app/v2box-v2ray-client/id6446814690)

**2.** Copy to clipboard:

-   [ ] **Always up-to-date**

    > Recommended Lists: **[6](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/6.txt)**, **[22](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/22.txt)**, **[23](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/23.txt)**, **[24](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/24.txt)** and **[25](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/25.txt)**.

    1.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/1.txt`
    2.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/2.txt`
    3.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/3.txt`
    4.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/4.txt`
    5.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/5.txt`
    6.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/6.txt`
    7.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/7.txt`
    8.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/8.txt`
    9.  `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/9.txt`
    10. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/10.txt`
    11. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/11.txt`
    12. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/12.txt`
    13. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/13.txt`
    14. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/14.txt`
    15. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/15.txt`
    16. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/16.txt`
    17. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/17.txt`
    18. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/18.txt`
    19. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/19.txt`
    20. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/20.txt`
    21. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/21.txt`
    22. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/22.txt`
    23. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/23.txt`
    24. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/24.txt`
    25. `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/25.txt`

**3.** Go to the **"V2Box - V2ray Client"** app and go to the **"Config"** tab, tap the plus icon in the top right corner, then — **"Add subscription"**, enter any **"Name"** and paste the configuration link into the **"URL"** field.

**4.** After adding the configuration, wait for the verification to finish and select the desired one, just by clicking on its name.

**5.** In the bottom panel of the program, click the **