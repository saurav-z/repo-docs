<!-- Social Media Links (moved to the top for better visibility) -->
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

# Goida VPN Configs: Your Source for Always-Up-to-Date VPN Configurations

This repository provides a continuously updated collection of public VPN configurations to bypass internet restrictions, ensuring reliable and easy access to the open web.  Check out the [original repository](https://github.com/AvenCores/goida-vpn-configs) for more details!

---

## Key Features

*   **Always Fresh:** Configurations are automatically updated every 9 minutes via GitHub Actions.
*   **Broad Compatibility:**  Works with a wide range of VPN clients, including `v2rayNG`, `NekoRay`, `Throne`, and many more.
*   **Multiple Protocols:** Supports `V2Ray`, `VLESS`, `Trojan`, `VMess`, `Reality`, and `Shadowsocks` configurations.
*   **Easy to Use:** Simple setup with clear instructions and readily available configuration links.
*   **QR Code Support:**  QR codes available for quick configuration import on Android TV.
*   **Comprehensive Guides:** Detailed guides for various platforms (Android, Android TV, iOS, Windows, macOS, Linux).

---

## Quick Start Guide

1.  **Choose a Configuration:** Select a link from the "**📋 Общий список всех вечно актуальных конфигов**" section below.
2.  **Import into your VPN Client:** Follow the instructions for your chosen VPN client (see guides below).
3.  **Connect:** Select a server with the lowest ping and connect!

---

## Contents

*   [Key Features](#key-features)
*   [Quick Start Guide](#quick-start-guide)
*   [How It Works](#how-it-works)
*   [Repository Structure](#repository-structure)
*   [Local Generator Run](#local-generator-run)
*   [Video Guides](#-видео-гайд-по-установке-и-решению-проблем)
*   [Platform-Specific Guides](#-общее-меню-гайдов-репозитория)
*   [License](#-лицензия)
*   [Support the Author](#-поддержать-автора)

---

## How It Works

*   The [`source/main.py`](source/main.py) script fetches public subscription links from various sources.
*   The  [`frequent_update.yml`](.github/workflows/frequent_update.yml)  workflow runs the script every 9 minutes using a cron job.
*   Results are stored in the  `githubmirror/`  directory and pushed back to the repository.

Each update generates a commit with a message like:
> 🚀 Обновление конфига по часовому поясу Европа/Москва: HH:MM | DD.MM.YYYY

---

## Repository Structure

```text
githubmirror/        — Generated .txt configs (25 files)
qr-codes/            — PNG config QR codes for import
source/              — Python script and dependencies
 ├─ main.py
 └─ requirements.txt
.github/workflows/   — CI/CD (automatic updates every 9 min)
README.md            — This file
```

---

## Local Generator Run

1.  `git clone https://github.com/AvenCores/goida-vpn-configs`
2.  `cd goida-vpn-configs/source`
3.  `python -m pip install -r requirements.txt`
4.  `export MY_TOKEN=<GITHUB_TOKEN>` (A GitHub token with repo access is required to push changes.)
5.  `python main.py` (Configs will be created in `../githubmirror`)

> **Important:**  If you are running the script from a fork, manually set `REPO_NAME = "<username>/<repository>"` in `source/main.py`.

---

# Video Guides
  
![maxresdefault](https://github.com/user-attachments/assets/e36e2351-3b1a-4b90-87f7-cafbc74f238c)
<div align="center">
> ⚠️ **Attention!** The video guide is for Android, Android TV, Windows, Linux, and macOS. The text guide below is the only option for iOS and iPadOS.

[**Watch on YouTube**](https://youtu.be/sagz2YluM70)  

[**Watch on Dzen**](https://dzen.ru/video/watch/680d58f28c6d3504e953bd6d)  

[**Watch on VK Video**](https://vk.com/video-200297343_456239303)

[**Watch on Telegram**](https://t.me/avencoreschat/56595)
</div>
---

# Platform-Specific Guides

<details>
<summary>👩‍💻 Source Code for generating the ever-current configurations</summary>

Link to the source code - [Link](https://github.com/AvenCores/goida-vpn-configs/tree/main/source)
</details>

---
<details>
<summary>📋 List of all always-up-to-date configurations</summary>

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

🔗 [Link to QR codes for ever-relevant configurations](https://github.com/AvenCores/goida-vpn-configs/tree/main/qr-codes)
</details>

---

<details>
<summary>📱 Guide for Android</summary>

**1.** Download **"v2rayNG"** - [Link](https://github.com/2dust/v2rayNG/releases/download/1.10.19/v2rayNG_1.10.19_universal.apk)

**2.** Copy to clipboard:

 - [ ] **Ever-relevant**

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

**3.** Go to the "v2rayNG" app and click the ➕ in the upper right corner, then select **"Import from clipboard"**.

**4.** Click **"the three dots in the upper right"**, then **"Check group profiles"**, after the check is complete, click **"Sort by test results"** in the same menu.

**5.** Select the server you need and then click the ▶️ button in the lower right corner.
</details>

---
<details>
<summary>📺 Guide for Android TV</summary>

**1.** Download **"v2rayNG"** - [Link](https://github.com/2dust/v2rayNG/releases/download/1.10.19/v2rayNG_1.10.19_universal.apk)

> Recommended **"QR codes"**: **[6](https://github.com/AvenCores/goida-vpn-configs/blob/main/qr-codes/6.png)**, **[22](https://github.com/AvenCores/goida-vpn-configs/blob/main/qr-codes/22.png)**, **[23](https://github.com/AvenCores/goida-vpn-configs/blob/main/qr-codes/23.png)**, **[24](https://github.com/AvenCores/goida-vpn-configs/blob/main/qr-codes/24.png)** and **[25](https://github.com/AvenCores/goida-vpn-configs/blob/main/qr-codes/25.png)**.

**2.** Download **"QR codes"** of always-up-to-date configurations - [Link](https://github.com/AvenCores/goida-vpn-configs/tree/main/qr-codes)

**3**. Go to the "v2rayNG" app and click the ➕ in the upper right corner, then select **"Import from QR code"**, select the picture by clicking on the photo icon in the upper right corner.

**4.** Click **"the three dots in the upper right"**, then **"Check group profiles"**, after the check is complete, click **"Sort by test results"** in the same menu.

**5.** Select the server you need and then click the ▶️ button in the lower right corner.
</details>

---
<details>
<summary>⚠ If there is no internet when connecting to VPN in v2rayNG</summary>

Link to a video demonstrating the fix - [Link](https://t.me/avencoreschat/25254)
</details>

---
<details>
<summary>⚠ If configurations do not appear when adding a VPN in v2rayNG</summary>

**1.** Click on the **"three bars"** in the **"upper left corner"**.

**2.** Click on the **"Groups"** button.

**3.** Click the **"circle with an arrow icon"** in the **"upper right corner"** and wait for the update to complete.
</details>

---
<details>
<summary>⚠ Fix error "Connection test failed: net/http: 12X handshake timeout"</summary>

**1.** On the desktop, press and hold the **"v2rayNG"** icon and click on the **"About App"** item.

**2.** Click the **"Stop"** button and restart **"v2rayNG"**.
</details>

---
<details>
<summary>⚠ Fix error "Fail to detect internet connection: io: read/write closed pipe"</summary>

**1.** On the desktop, press and hold the **"v2rayNG"** icon and click on the **"About App"** item.

**2.** Click the **"Stop"** button and restart **"v2rayNG"**.

**3.** Click **"the three dots in the upper right"**, then **"Check group profiles"**, after the check is complete, click **"Sort by test results"** in the same menu.

**4.** Select the server you need and then click the ▶️ button in the lower right corner.
</details>

---
<details>
<summary>🔄 Updating configurations in v2rayNG</summary>

**1.** Click on the **"three bars icon"** in the **"upper left corner"**.

**2.** Select the **"Groups"** tab.

**3.** Click on the **"circle with an arrow icon"** in the **"upper right corner"**.
</details>

---
<details>
<summary>🖥 Guide for Windows, Linux</summary>

**1.** Download **"Throne"** - [Windows 10/11](https://github.com/throneproj/Throne/releases/download/1.0.5/Throne-1.0.5-windows64.zip) / [Windows 7/8/8.1](https://github.com/throneproj/Throne/releases/download/1.0.5/Throne-1.0.5-windowslegacy64.zip) / [Linux](https://github.com/throneproj/Throne/releases/download/1.0.5/Throne-1.0.5-linux-amd64.zip)

**2.** Copy to clipboard:

 - [ ] **Ever-relevant**

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

**3.** Click on **"Profiles"**, then **"Add profile from clipboard"**.

**4.** Select all configurations with the key combination **"Ctrl + A"**, click **"Profiles"** in the top menu, then **"Latency test (ping) of the selected profile"** and wait for the test to complete (the text **"Latency test (ping) completed!"** will appear in the **"Logs"** tab)

**5.** Click the column button **"Latency (ping)"**.

**6.** At the top of the program window, activate the **"TUN mode"** option by checking the box.

**7.** Select one of the configurations with the lowest **"Latency (ping)"**, then click **"LMB"** and **"Run"**.
</details>

---
<details>
<summary>⚠ Correcting the MSVCP and VCRUNTIME error on Windows 10/11</summary>

**1.** Press **"Win+R"** and type **"control"**.

**2.** Select **"Programs and Features"**.

**3.** In the search (top right) type the word **"Visual"** and remove everything related to **"Microsoft Visual"**.

**4.** Download and unzip the archive - [Link](https://cf.comss.org/download/Visual-C-Runtimes-All-in-One-Jul-2025.zip)

**5.** Run **"install_bat.all"** from *Administrator* and wait until everything is installed.
</details>

---
<details>
<summary>🔄 Updating configurations in NekoRay</summary>

**1.** Click the **"Settings"** button.

**2.** Select **"Groups"**.

**3.** Click the **"Update all subscriptions"** button.
</details>

---
<details>
<summary>☎ Guide for iOS, iPadOS</summary>

**1.** Download **"V2Box - V2ray Client"** - [Link](https://apps.apple.com/ru/app/v2box-v2ray-client/id6446814690)

**2.** Copy to clipboard:

 - [ ] **Ever-relevant**

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

**3.** Go to the "V2Box - V2ray Client" app and go to the "Config" tab, click the plus sign in the upper right corner, then - **"Add subscription"**, enter any **"Name"** and paste the configuration link in the **"URL"** field.

**4.** After adding the configuration, wait for the check to complete and select the desired one by simply clicking on its name.

**5.** In the bottom panel of the program, click the **"Connect"** button.
</details>

---
<details>
<summary>🔄 Updating configurations in V2Box - V2ray Client</summary>

**1.** Go to the **"Config"** tab