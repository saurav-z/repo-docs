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

# Goida VPN Configs: Get Free and Always Up-to-Date VPN Configurations

This repository provides a regularly updated collection of public VPN configurations for bypassing internet restrictions, offering you secure and private access to the web.  [View the original repository](https://github.com/AvenCores/goida-vpn-configs).

## Key Features

*   **Automatic Updates:** Configurations are refreshed every 9 minutes using GitHub Actions, ensuring you always have access to the latest, working VPN settings.
*   **Multiple Protocols:** Supports a variety of VPN protocols including V2Ray, VLESS, Trojan, VMess, Reality, and Shadowsocks, providing flexibility for different client applications.
*   **Wide Compatibility:** Compatible with popular VPN clients such as v2rayNG, NekoRay, Throne, v2rayN, V2Box, v2RayTun, Hiddify, and more.
*   **Easy to Use:** Simply copy a link and import it into your preferred VPN client for instant access.
*   **QR Code Support:**  Quickly import configurations using QR codes for Android TV and other devices.
*   **Comprehensive Guides:** Detailed guides and video tutorials are available for Android, iOS, Windows, MacOS, and Linux, making setup easy.

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

1.  Copy the desired link from the "General List of Always Up-to-Date Configurations" section.
2.  Import it into your chosen VPN client (see instructions below).
3.  Select a server with the lowest ping and connect.

---

## How It Works

*   The script `source/main.py` downloads public subscriptions from various sources.
*   The `frequent_update.yml` workflow runs the script using cron every 9 minutes.
*   Results are saved in the `githubmirror/` directory and pushed to this repository.

Each run creates a commit of the following type:

> 🚀 Config update by the time zone Europe/Moscow: HH:MM | DD.MM.YYYY

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

>   **Important!** In the `source/main.py` file, manually set `REPO_NAME = "<username>/<repository>"` if you are running the script from a fork.

---

# Video Guides

![maxresdefault](https://github.com/user-attachments/assets/e36e2351-3b1a-4b90-87f7-cafbc74f238c)

<div align="center">

> ⚠️ **Important!** Only the text guide below is relevant for iOS and iPadOS. The video guide is only valid for Android, Android TV, Windows, Linux, and MacOS.

[**Watch on YouTube**](https://youtu.be/sagz2YluM70)

[**Watch on Dzen**](https://dzen.ru/video/watch/680d58f28c6d3504e953bd6d)

[**Watch on VK Video**](https://vk.com/video-200297343_456239303)

[**Watch on Telegram**](https://t.me/avencoreschat/56595)

</div>

---

# Guide Menu

<details>
  <summary>Source Code for Generating Always Up-to-Date Configurations</summary>
  Link to source code — [Link](https://github.com/AvenCores/goida-vpn-configs/tree/main/source)
</details>

---

<details>
  <summary>General List of Always Up-to-Date Configurations</summary>

  > Recommended Lists: **[6](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/6.txt)**, **[22](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/22.txt)**, **[23](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/23.txt)**, **[24](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/24.txt)**, and **[25](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/25.txt)**.

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

  🔗 [Link to QR Codes of Always Up-to-Date Configurations](https://github.com/AvenCores/goida-vpn-configs/tree/main/qr-codes)
</details>

---

<details>
  <summary>📱 Guide for Android</summary>

  **1.** Download **"v2rayNG"** — [Link](https://github.com/2dust/v2rayNG/releases/download/1.10.19/v2rayNG_1.10.19_universal.apk)

  **2.** Copy to clipboard:

  -   [ ] **Always Up-to-Date**

  >   Recommended Lists: **[6](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/6.txt)**, **[22](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/22.txt)**, **[23](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/23.txt)**, **[24](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/24.txt)**, and **[25](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/25.txt)**.

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

  **3.** Go to the **"v2rayNG"** app and click the ➕ in the top right corner, then select **"Import from clipboard"**.

  **4.** Click **"the three dots in the top right"**, and then **"Test group profiles"**, after the test, select **"Sort by test results"** in the same menu.

  **5.** Select the server you want and then click the ▶️ button in the bottom right corner.
</details>

---

<details>
  <summary>📺 Guide for Android TV</summary>

  **1.** Download **"v2rayNG"** — [Link](https://github.com/2dust/v2rayNG/releases/download/1.10.19/v2rayNG_1.10.19_universal.apk)

  > Recommended **"QR Codes"**: **[6](https://github.com/AvenCores/goida-vpn-configs/blob/main/qr-codes/6.png)**, **[22](https://github.com/AvenCores/goida-vpn-configs/blob/main/qr-codes/22.png)**, **[23](https://github.com/AvenCores/goida-vpn-configs/blob/main/qr-codes/23.png)**, **[24](https://github.com/AvenCores/goida-vpn-configs/blob/main/qr-codes/24.png)**, and **[25](https://github.com/AvenCores/goida-vpn-configs/blob/main/qr-codes/25.png)**.

  **2.** Download the **"QR Codes"** of the always up-to-date configurations — [Link](https://github.com/AvenCores/goida-vpn-configs/tree/main/qr-codes)

  **3.** Go to the **"v2rayNG"** app and click the ➕ in the top right corner, then select **"Import from QR code"**, select the picture by clicking the photo icon in the top right corner.

  **4.** Click **"the three dots in the top right"**, and then **"Test group profiles"**, after the test, select **"Sort by test results"** in the same menu.

  **5.** Select the server you want and then click the ▶️ button in the bottom right corner.
</details>

---

<details>
  <summary>⚠ If there is no internet connection when connecting to VPN in v2rayNG</summary>
  Link to the video with the fix demonstration — [Link](https://t.me/avencoreschat/25254)
</details>

---

<details>
  <summary>⚠ If the configurations did not appear when adding a VPN to v2rayNG</summary>

  **1.** Click on the **"three lines"** in the **"upper left corner"**.

  **2.** Click on the **"Groups"** button.

  **3.** Click on the **"circle with an arrow icon"** in the **"upper right corner"** and wait for the update to complete.
</details>

---

<details>
  <summary>⚠ Fix the error "Connection test failed"</summary>
  **1.** Select the "Profile test"
  **2.** After testing, select "Sort by test results" in the same menu
  **3.** Try to select the server that has the best test results and then click the ▶️ button in the bottom right corner.
</details>

---

<details>
  <summary>🔄 Updating configurations in v2rayNG</summary>

  **1.** Click on the **"three lines icon"** in the **"upper left corner"**.

  **2.** Select the **"Groups"** tab.

  **3.** Click on the **"circle with an arrow icon"** in the **"upper right corner"**.
</details>

---

<details>
  <summary>🖥 Guide for Windows, Linux</summary>

  **1.** Download **"Throne"** — [Windows 10/11](https://github.com/throneproj/Throne/releases/download/1.0.6/Throne-1.0.6-windows64.zip) / [Windows 7/8/8.1](https://github.com/throneproj/Throne/releases/download/1.0.6/Throne-1.0.6-windowslegacy64.zip) / [Linux](https://github.com/throneproj/Throne/releases/download/1.0.6/Throne-1.0.6-linux-amd64.zip)

  **2.** Copy to clipboard:

  -   [ ] **Always Up-to-Date**

  >   Recommended Lists: **[6](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/6.txt)**, **[22](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/22.txt)**, **[23](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/23.txt)**, **[24](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/24.txt)**, and **[25](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/25.txt)**.

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

  **3.** Click on **"Profiles"**, and then **"Add profile from clipboard"**.

  **4.** Select all configurations with the **"Ctrl + A"** key combination, click **"Profiles"** in the top menu, and then **"Latency (ping) test of the selected profile"** and wait for the test to complete (the message **"Latency (ping) test completed!"** will appear in the **"Logs"** tab)

  **5.** Click on the column button **"Latency (ping)"**.

  **6.** In the upper part of the program, activate the option **"TUN Mode"** by checking the box.

  **7.** Select one of the configurations with the smallest **"Latency (ping)"**, then click **"LMB"** and **"Start"**.
</details>

---

<details>
  <summary>⚠ Correcting the MSVCP and VCRUNTIME error on Windows 10/11</summary>

  **1.** Press **"Win+R"** and write **"control"**.

  **2.** Select **"Programs and features"**.

  **3.** In the search bar (top right), type the word **"Visual"** and delete everything related to **"Microsoft Visual"**.

  **4.** Download the archive and unpack it — [Link](https://cf.comss.org/download/Visual-C-Runtimes-All-in-One-Jul-2025.zip)

  **5.** Run **"install_bat.all"** from *Administrator* and wait for everything to install.
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

  **1.** Download **"V2Box - V2ray Client"** — [Link](https://apps.apple.com/ru/app/v2box-v2ray-client/id6446814690)

  **2.** Copy to clipboard:

  -   [ ] **Always Up-to-Date**

  >   Recommended Lists: **[6](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/6.txt)**, **[22](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/22.txt)**, **[23](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/23.txt)**, **[24](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/24.txt)**, and **[25](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/25.txt)**.

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

  **3.** Go to the **"V2Box - V2ray Client"** app and go to the **"Config"** tab, click the plus sign in the upper right corner, then — **"Add subscription"**, enter any **"Name"** and paste the link to the config in the **"URL"** field.

  **4.** After adding the config, wait for the test to complete and select the one you need by simply clicking on its name.

  **5.** Click the **"Connect"** button in the bottom panel of the program.
</details>

---

<details>
  <summary>🔄 Updating configurations in V2Box - V2ray Client</summary>

  **1.** Go to the **"Config"** tab.

  **2.** Click on the refresh icon to the left of the subscription group name.
</