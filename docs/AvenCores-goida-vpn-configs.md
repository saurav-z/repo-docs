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

# Goida VPN Configs: Stay Connected with Always-Up-to-Date VPN Configurations!

Access a constantly updated collection of public VPN configurations for V2Ray, VLESS, Trojan, VMess, Reality, and Shadowsocks, bypassing online restrictions effortlessly.  Find the original repository [here](https://github.com/AvenCores/goida-vpn-configs).

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://github.com/AvenCores/goida-vpn-configs)
[![GPL-3.0 License](https://img.shields.io/badge/License-GPL--3.0-blue?style=for-the-badge)](./LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/AvenCores/goida-vpn-configs?style=for-the-badge)](https://github.com/AvenCores/goida-vpn-configs/issues)
[![GitHub stars](https://img.shields.io/github/stars/AvenCores/goida-vpn-configs?style=for-the-badge)](https://github.com/AvenCores/goida-vpn-configs/stargazers)
![GitHub forks](https://img.shields.io/github/forks/AvenCores/goida-vpn-configs?style=for-the-badge)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/AvenCores/goida-vpn-configs?style=for-the-badge)](https://github.com/AvenCores/goida-vpn-configs/pulls)

## Key Features:

*   **Always Fresh:** Configurations updated every 9 minutes via GitHub Actions.
*   **Broad Compatibility:** Works with popular VPN clients like v2rayNG, NekoRay, Throne, and more.
*   **Multiple Protocols:** Supports V2Ray, VLESS, Trojan, VMess, Reality, and Shadowsocks.
*   **Easy to Use:** Simple copy-paste setup with readily available configuration links.
*   **Regular Updates:**  Ensure you have the latest, working configurations.

## Quick Start:

1.  **Choose a Config:** Select a link from the "[📋 Общий список всех вечно актуальных конфигов](#-общий-список-всех-вечно-актуальных-конфигов)" section.
2.  **Import into Client:**  Import the link into your preferred VPN client.
3.  **Connect:**  Select a server with minimal ping and connect!

## How It Works:

*   The Python script (`source/main.py`) fetches public subscriptions from various sources.
*   A GitHub Actions workflow (`frequent_update.yml`) runs the script every 9 minutes via cron.
*   Results are saved in the `githubmirror/` directory and pushed to this repository.
*   Each run creates a commit with the format: "🚀 Обновление конфига по часовому поясу Европа/Москва: HH:MM | DD.MM.YYYY"

## Repository Structure:

```text
githubmirror/        — сгенерированные .txt конфиги (23 файла)
qr-codes/            — PNG-версии конфигов для импорта по QR
source/              — Python-скрипт и зависимости генератора
 ├─ main.py
 └─ requirements.txt
.github/workflows/   — CI/CD (авто-обновление каждые 9 мин)
README.md            — этот файл
```

## Local Generator Run:

```bash
git clone https://github.com/AvenCores/goida-vpn-configs
cd goida-vpn-configs/source
python -m pip install -r requirements.txt
export MY_TOKEN=<GITHUB_TOKEN>   # токен с правом repo, чтобы пушить изменения
python main.py                  # конфиги появятся в ../githubmirror
```

> **Important:**  If running from a fork, manually set `REPO_NAME = "<username>/<repository>"` in `source/main.py`.

## Video Guide:

**[Watch Video Guide (YouTube)](https://youtu.be/sagz2YluM70)**

**[Watch Video Guide (Dzen)](https://dzen.ru/video/watch/680d58f28c6d3504e953bd6d)**

**[Watch Video Guide (VK Video)](https://vk.com/video-200297343_456239303)**

**[Watch Video Guide (Telegram)](https://t.me/avencoreschat/56595)**

> ⚠️ **Note:** The video guide is only applicable to Android, Android TV, Windows, Linux, and macOS. The text guide below is relevant for iOS and iPadOS.

## Guide Menu:

<details>
<summary>👩‍💻 Source Code</summary>
Link to the source code — [Link](https://github.com/AvenCores/goida-vpn-configs/tree/main/source)
</details>

<details>
<summary>📋 Always Up-to-Date Config Lists</summary>
> Recommended Lists: **[6](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/6.txt)**, **[22](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/22.txt)**, **[23](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/23.txt)**, **[24](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/24.txt)** и **[25](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/25.txt)**.

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

🔗 [QR Codes Link](https://github.com/AvenCores/goida-vpn-configs/tree/main/qr-codes)
</details>

<details>
<summary>📱 Android Guide</summary>
**1.** Download **«v2rayNG»** — [Link](https://github.com/2dust/v2rayNG/releases/download/1.10.19/v2rayNG_1.10.19_universal.apk)

**2.** Copy to clipboard:

 - [ ] **Always up-to-date**

> Recommended Lists: **[6](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/6.txt)**, **[22](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/22.txt)**, **[23](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/23.txt)**, **[24](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/24.txt)** и **[25](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/25.txt)**.

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

**3.** In the **«v2rayNG»** app, tap the ➕ in the top right corner and select **«Import from clipboard»**.

**4.** Tap the **«three dots»** in the top right corner, then **«Check group profiles»**, and after checking, select **«Sort by test results»** in the same menu.

**5.** Choose the server with the lowest ping and then click the ▶️ button in the bottom right corner.
</details>

<details>
<summary>📺 Android TV Guide</summary>
**1.** Download **«v2rayNG»** — [Link](https://github.com/2dust/v2rayNG/releases/download/1.10.19/v2rayNG_1.10.19_universal.apk)

> Recommended **«QR-codes»**: **[6](https://github.com/AvenCores/goida-vpn-configs/blob/main/qr-codes/6.png)**, **[22](https://github.com/AvenCores/goida-vpn-configs/blob/main/qr-codes/22.png)**, **[23](https://github.com/AvenCores/goida-vpn-configs/blob/main/qr-codes/23.png)**, **[24](https://github.com/AvenCores/goida-vpn-configs/blob/main/qr-codes/24.png)** и **[25](https://github.com/AvenCores/goida-vpn-configs/blob/main/qr-codes/25.png)**.

**2.** Download **«QR-codes»** — [Link](https://github.com/AvenCores/goida-vpn-configs/tree/main/qr-codes)

**3**. In the **«v2rayNG»** app, tap the ➕ in the top right corner and select **«Import from QR-code»**, select the picture by clicking on the photo icon in the upper right corner.

**4.** Tap the **«three dots»** in the top right corner, then **«Check group profiles»**, and after checking, select **«Sort by test results»** in the same menu.

**5.** Choose the server you need and then click the ▶️ button in the bottom right corner.
</details>

<details>
<summary>⚠ If there is no internet when connecting to VPN in v2rayNG</summary>
Link to the video demonstration of the fix — [Link](https://t.me/avencoreschat/25254)
</details>

<details>
<summary>⚠ If configs do not appear when adding VPN to v2rayNG</summary>

**1.** Click on the **«three stripes»** in the **«upper left corner»**.

**2.** Click on the **«Groups»** button.

**3.** Click on the **«circle icon with an arrow»** in the **«upper right corner»** and wait for the update to finish.
</details>

<details>
<summary>⚠ Fix "Connection check failure: net/http: 12X handshake timeout"</summary>
**1.** On the desktop, hold the **«v2rayNG»** icon and click on the **«About the application»** item.

**2.** Click the **«Stop»** button and restart **«v2rayNG»**.
</details>

<details>
<summary>⚠ Fix "Fail to detect internet connection: io: read/write closed pipe"</summary>

**1.** On the desktop, hold the **«v2rayNG»** icon and click on the **«About the application»** item.

**2.** Click the **«Stop»** button and restart **«v2rayNG»**.

**3.** Tap the **«three dots»** in the top right corner, then **«Check group profiles»**, and after checking, select **«Sort by test results»** in the same menu.

**4.** Choose the server you need and then click the ▶️ button in the bottom right corner.
</details>

<details>
<summary>🔄 Updating configs in v2rayNG</summary>
**1.** Click on the **«icon with three stripes»** in the **«upper left corner»**.

**2.** Select the **«Groups»** tab.

**3.** Click on the **«icon with a circle with an arrow»** in the **«upper right corner»**.
</details>

<details>
<summary>🖥 Windows, Linux Guide</summary>
**1.** Download **«Throne»** — [Windows 10/11](https://github.com/throneproj/Throne/releases/download/1.0.5/Throne-1.0.5-windows64.zip) / [Windows 7/8/8.1](https://github.com/throneproj/Throne/releases/download/1.0.5/Throne-1.0.5-windowslegacy64.zip) / [Linux](https://github.com/throneproj/Throne/releases/download/1.0.5/Throne-1.0.5-linux-amd64.zip)

**2.** Copy to clipboard:

 - [ ] **Always up-to-date**

> Recommended Lists: **[6](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/6.txt)**, **[22](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/22.txt)**, **[23](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/23.txt)**, **[24](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/24.txt)** и **[25](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/25.txt)**.

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

**3.** Click on **«Profiles»**, then **«Add profile from clipboard»**.

**4.** Select all configs with the key combination **«Ctrl + A»**, click **«Profiles»** in the top menu, then **«Latency (ping) test of the selected profile»** and wait for the test to finish (the inscription **«Latency (ping) test completed!»** will appear in the **«Logs»** tab)

**5.** Click on the **«Latency (ping)»** column.

**6.** At the top of the program window, activate the **«TUN mode»** option by checking the box.

**7.** Select one of the configs with the lowest **«Latency (ping)»**, then click **«LMB»** and **«Run»**.
</details>

<details>
<summary>⚠ Fix MSVCP and VCRUNTIME errors on Windows 10/11</summary>

**1.** Press **«Win+R»** and write **«control»**.

**2.** Select **«Programs and Features»**.

**3.** In the search (top right) type the word **«Visual»** and delete everything related to **«Microsoft Visual»**.

**4.** Download and unpack the archive — [Link](https://cf.comss.org/download/Visual-C-Runtimes-All-in-One-Jul-2025.zip)

**5.** Run **«install_bat.all»** *as Administrator* and wait for everything to install.
</details>

<details>
<summary>🔄 Updating configs in NekoRay</summary>
**1.** Click on the **«Settings»** button.

**2.** Select **«Groups»**.

**3.** Click on the **«Update all subscriptions»** button.
</details>

<details>
<summary>☎ iOS, iPadOS Guide</summary>

**1.** Download **«V2Box - V2ray Client»** — [Link](https://apps.apple.com/ru/app/v2box-v2ray-client/id6446814690)

**2.** Copy to clipboard:

 - [ ] **Always up-to-date**

> Recommended Lists: **[6](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/6.txt)**, **[22](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/22.txt)**, **[23](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/23.txt)**, **[24](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/24.txt)** и **[25](https://github.com/AvenCores/goida-vpn-configs/raw/refs/heads/main/githubmirror/25.txt)**.

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

**3.** In the **«V2Box - V2ray Client»** app, go to the **«Config»** tab, click the plus sign in the upper right corner, then - **«Add subscription»**, enter any **«Name»** and paste the config link into the **«URL»** field.

**4.** After adding the config, wait for the check to finish and select the one you want by simply clicking on its name.

**5.** Press the **«Connect»** button in the bottom panel of the program.
</details>

<details>
<summary>🔄 Updating configs in V2Box - V2ray Client</summary>
**1.** Go to the **«Config»** tab.

**2.** Click on the update icon to the left