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

# Get Unlimited Access: Free, Updated VPN Configs for Secure Internet Access

Access a constantly updated collection of free VPN configurations using V2Ray, VLESS, Trojan, VMess, Reality, and Shadowsocks protocols.  Find the original repository [here](https://github.com/AvenCores/goida-vpn-configs).

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://github.com/AvenCores/goida-vpn-configs)
[![GPL-3.0 License](https://img.shields.io/badge/License-GPL--3.0-blue?style=for-the-badge)](./LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/AvenCores/goida-vpn-configs?style=for-the-badge)](https://github.com/AvenCores/goida-vpn-configs/issues)
[![GitHub stars](https://img.shields.io/github/stars/AvenCores/goida-vpn-configs?style=for-the-badge)](https://github.com/AvenCores/goida-vpn-configs/stargazers)
![GitHub forks](https://img.shields.io/github/forks/AvenCores/goida-vpn-configs?style=for-the-badge)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/AvenCores/goida-vpn-configs?style=for-the-badge)](https://github.com/AvenCores/goida-vpn-configs/pulls)

## Key Features

*   **Automatic Updates:**  Configs are updated every 9 minutes using GitHub Actions, ensuring fresh and reliable connections.
*   **Wide Compatibility:**  Works with popular VPN clients like `v2rayNG`, `NekoRay`, `Throne`, `v2rayN`, `V2Box`, `v2RayTun`, and `Hiddify`.
*   **Multiple Protocol Support:**  Includes configurations for `V2Ray`, `VLESS`, `Trojan`, `VMess`, `Reality`, and `Shadowsocks`.
*   **Easy Import:** Configurations are provided as simple text subscriptions for easy import into your VPN client.
*   **QR Code Support:** QR codes are also provided for easy setup on Android TV.

## Table of Contents

*   [Key Features](#key-features)
*   [Quick Start](#quick-start)
*   [How It Works](#how-it-works)
*   [Repository Structure](#repository-structure)
*   [Local Generator Run](#local-generator-run)
*   [Video Guides](#video-guides)
*   [Guide Menu](#guide-menu)
*   [License](#license)
*   [Support the Author](#support-the-author)

---

## Quick Start

1.  Copy a link from the "Always Up-to-Date Config List" section below.
2.  Import the link into your VPN client.
3.  Select a server with low ping and connect.

---

## How It Works

*   The [`source/main.py`](source/main.py) script downloads public subscriptions from various sources.
*   The [`frequent_update.yml`](.github/workflows/frequent_update.yml) workflow runs the script every 9 minutes using a cron job.
*   Results are saved to the `githubmirror/` directory and pushed to this repository.

Each update creates a commit like this:
> 🚀 Config update for Europe/Moscow timezone: HH:MM | DD.MM.YYYY

---

## Repository Structure

```text
githubmirror/        — Generated .txt config files (23 files)
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

> **Important!**  In `source/main.py`, set `REPO_NAME = "<username>/<repository>"` if you're running the script from a fork.

---

## Video Guides

![maxresdefault](https://github.com/user-attachments/assets/e36e2351-3b1a-4b90-87f7-cafbc74f238c)

<div align="center">

> ⚠️ **Attention!** The video guide is only relevant for Android, Android TV, Windows, Linux, and MacOS. For iOS and iPadOS, refer to the text guide below.

[**Watch on YouTube**](https://youtu.be/sagz2YluM70)

[**Watch on Dzen**](https://dzen.ru/video/watch/680d58f28c6d3504e953bd6d)

[**Watch on VK Video**](https://vk.com/video-200297343_456239303)

[**Watch on Telegram**](https://t.me/avencoreschat/56595)

</div>

---

## Guide Menu

<details>
  <summary>👩‍💻 Source Code</summary>
  Link to the source code: [Source Code](https://github.com/AvenCores/goida-vpn-configs/tree/main/source)
</details>

---

<details>
  <summary>📋 Always Up-to-Date Config List</summary>
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

🔗 [QR Codes for Always Up-to-Date Configs](https://github.com/AvenCores/goida-vpn-configs/tree/main/qr-codes)
</details>

---

<details>
  <summary>📱 Android Guide</summary>
    **1.** Download **«v2rayNG»** — [Download Link](https://github.com/2dust/v2rayNG/releases/download/1.10.19/v2rayNG_1.10.19_universal.apk)

    **2.** Copy to clipboard:

    - [ ] **Always Up-to-Date**

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

    **3.** Open the **«v2rayNG»** app, tap the ➕ in the upper right corner, and select **«Import from clipboard»**.

    **4.** Tap **«three dots»** in the upper right corner, and then **«Test group profiles»**. After testing, tap **«Sort by test results»** in the same menu.

    **5.** Choose your preferred server, then tap the ▶️ button in the lower right corner.
</details>

---

<details>
  <summary>📺 Android TV Guide</summary>
    **1.** Download **«v2rayNG»** — [Download Link](https://github.com/2dust/v2rayNG/releases/download/1.10.19/v2rayNG_1.10.19_universal.apk)

    > Recommended **«QR Codes»**: **[6](https://github.com/AvenCores/goida-vpn-configs/blob/main/qr-codes/6.png)**, **[22](https://github.com/AvenCores/goida-vpn-configs/blob/main/qr-codes/22.png)**, **[23](https://github.com/AvenCores/goida-vpn-configs/blob/main/qr-codes/23.png)**, **[24](https://github.com/AvenCores/goida-vpn-configs/blob/main/qr-codes/24.png)** and **[25](https://github.com/AvenCores/goida-vpn-configs/blob/main/qr-codes/25.png)**.

    **2.** Download the **«QR Codes»** for the always up-to-date configurations — [Link](https://github.com/AvenCores/goida-vpn-configs/tree/main/qr-codes)

    **3**. Open the **«v2rayNG»** app, tap the ➕ in the upper right corner, and select **«Import from QR code»**. Then, select the image by tapping the photo icon in the upper right corner.

    **4.** Tap **«three dots»** in the upper right corner, and then **«Test group profiles»**. After testing, tap **«Sort by test results»** in the same menu.

    **5.** Choose your preferred server, then tap the ▶️ button in the lower right corner.
</details>

---

<details>
  <summary>⚠ If you have no internet connection when connecting to VPN in v2rayNG</summary>
  Link to the video demonstrating the fix — [Link](https://t.me/avencoreschat/25254)
</details>

---

<details>
  <summary>⚠ If configurations do not appear when adding VPN in v2rayNG</summary>
    **1.** Tap the **«three lines»** in the **«upper left corner»**.

    **2.** Tap the **«Groups»** button.

    **3.** Tap the **«circle with an arrow icon»** in the **«upper right corner»** and wait for the update to finish.
</details>

---

<details>
  <summary>⚠ Fix for "Internet connection check failed: net/http: 12X handshake timeout"</summary>
    **1.** Press and hold the **«v2rayNG»** icon on your desktop and tap **«About app»**.

    **2.** Tap the **«Stop»** button and restart **«v2rayNG»**.
</details>

---

<details>
  <summary>⚠ Fix for "Fail to detect internet connection: io: read/write closed pipe"</summary>
    **1.** Press and hold the **«v2rayNG»** icon on your desktop and tap **«About app»**.

    **2.** Tap the **«Stop»** button and restart **«v2rayNG»**.

    **3.** Tap **«three dots»** in the upper right corner, and then **«Test group profiles»**. After testing, tap **«Sort by test results»** in the same menu.

    **4.** Choose your preferred server, then tap the ▶️ button in the lower right corner.
</details>

---

<details>
  <summary>🔄 Updating configurations in v2rayNG</summary>
    **1.** Tap the **«three lines icon»** in the **«upper left corner»**.

    **2.** Select the **«Groups»** tab.

    **3.** Tap the **«circle with an arrow icon»** in the **«upper right corner»**.
</details>

---

<details>
  <summary>🖥 Windows, Linux Guide</summary>
    **1.** Download **«Throne»** — [Windows 10/11](https://github.com/throneproj/Throne/releases/download/1.0.5/Throne-1.0.5-windows64.zip) / [Windows 7/8/8.1](https://github.com/throneproj/Throne/releases/download/1.0.5/Throne-1.0.5-windowslegacy64.zip) / [Linux](https://github.com/throneproj/Throne/releases/download/1.0.5/Throne-1.0.5-linux-amd64.zip)

    **2.** Copy to clipboard:

    - [ ] **Always Up-to-Date**

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

    **3.** Click **«Profiles»**, then **«Add profile from clipboard»**.

    **4.** Select all configurations with the combination of keys **«Ctrl + A»**, click **«Profiles»** in the top menu, and then **«Latency test (ping) of the selected profile»** and wait for the test to complete (the inscription **«Latency test (ping) complete!»** will appear in the **«Logs»** tab)

    **5.** Click on the column button **«Latency (ping)»**.

    **6.** In the upper part of the program window, activate the **«TUN mode»** option by checking the box.

    **7.** Select one of the configurations with the lowest **«Latency (ping)»**, then click **«LMB»** and **«Start»**.
</details>

---

<details>
  <summary>⚠ Fix MSVCP and VCRUNTIME error on Windows 10/11</summary>
    **1.** Press **«Win+R»** and write **«control»**.

    **2.** Select **«Programs and Features»**.

    **3.** In the search field (top right), type the word **«Visual»** and remove everything related to **«Microsoft Visual»**.

    **4.** Download the archive and unpack — [Link](https://cf.comss.org/download/Visual-C-Runtimes-All-in-One-Jul-2025.zip)

    **5.** Run **«install_bat.all»** *as an administrator* and wait for everything to install.
</details>

---

<details>
  <summary>🔄 Updating configurations in NekoRay</summary>
    **1.** Click the **«Settings»** button.

    **2.** Select **«Groups»**.

    **3.** Click the **«Update all subscriptions»** button.
</details>

---

<details>
  <summary>☎ iOS, iPadOS Guide</summary>
    **1.** Download **«V2Box - V2ray Client»** — [Download Link](https://apps.apple.com/ru/app/v2box-v2ray-client/id6446814690)

    **2.** Copy to clipboard:

    - [ ] **Always Up-to-Date**

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
    21) `https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/