[![GitHub last commit](https://img.shields.io/github/last-commit/barry-far/V2ray-Configs.svg)](https://github.com/Epodonios/v2ray-configs)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Update Configs](https://github.com/barry-far/V2ray-Configs/actions/workflows/main.yml/badge.svg)](https://github.com/Epodonios/V2ray-Configs/actions/workflows/main.yml)
[![GitHub repo size](https://img.shields.io/github/repo-size/Epodonios/V2ray-Configs)](https://github.com/Epodonios/v2ray-configs)

<a href="https://t.me/+IOG0nSifAV03ZmY0" target="_blank">
  <img src="https://cdn-icons-png.flaticon.com/512/2111/2111646.png" alt="Telegram" width="500" height="500"> Contact Us
</a>

# Access the Internet Securely and Anonymously with Free V2Ray Configurations

This repository, found at [https://github.com/Epodonios/v2ray-configs](https://github.com/Epodonios/v2ray-configs), provides a regularly updated collection of free V2Ray configurations. These configs can be used to securely and anonymously access the internet.

**Key Features:**

*   **Regularly Updated:** Configurations are collected and updated every five minutes.
*   **Multiple Protocols:** Supports Vmess, Vless, Trojan, Tuic, Shadowsocks, and ShadowsocksR.
*   **Subscription Links:** Provides subscription links for easy configuration in V2Ray clients.
*   **Multiple Formats:** Available in base64, normal, and split formats for versatility.
*   **Cross-Platform Compatibility:** Works with various V2Ray clients on Android, iOS, Windows, and Linux.

**Supported V2Ray Clients:**

*   **Android:** v2rayNG
*   **iOS:** fair, streisand
*   **Windows & Linux:** hiddify-next, nekoray, v2rayN

## Subscription Links

Use these subscription links to easily configure your V2Ray client:

*   **All Configs:**
    ```
    https://github.com/Epodonios/v2ray-configs/raw/main/All_Configs_Sub.txt
    ```
*   **Base64 Encoded (if the above fails):**
    ```
    https://github.com/Epodonios/v2ray-configs/raw/main/All_Configs_base64_Sub.txt
    ```

### Split by Protocol

*   **Vless:**
    ```
    https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/vless.txt
    ```
*   **Vmess:**
    ```
    https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/vmess.txt
    ```
*   **Shadowsocks (ss):**
    ```
    https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/ss.txt
    ```
*   **ShadowsocksR (ssr):**
    ```
    https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/ssr.txt
    ```
*   **Trojan:**
    ```
    https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/trojan.txt
    ```

### Configs Split into Batches of 250

*   Config List 1:
    ```
    https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub1.txt
    ```
*   Config List 2:
    ```
    https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub2.txt
    ```
*   ... (Config Lists 3-14 follow the same pattern, incrementing the number)

## How to Use

1.  Copy the provided subscription links.
2.  Go to your V2Ray client's subscription settings.
3.  Paste the link and save.
4.  Regularly update your subscriptions within your client.

**Enjoy secure and private internet access!**

---

## Advanced Usage: Tunneling Your Entire System

Here's how to use a proxy program to tunnel your entire system's traffic (like games):

### Using Proxifier

1.  **Install Proxifier:**  Download and install Proxifier from https://proxifier.com/download/.
2.  **Activate Proxifier:** Use one of these activation keys (Portable and Standard editions provided):

    *   **Portable Edition:** `L6Z8A-XY2J4-BTZ3P-ZZ7DF-A2Q9C`
    *   **Standard Edition:** `5EZ8G-C3WL5-B56YG-SCXM9-6QZAP`
    *   **Mac OS:** `P427L-9Y552-5433E-8DSR3-58Z68`

3.  **Add Proxy Server:** In Proxifier, go to "Profile" > "Proxy Servers" and click "Add".
4.  **Enter Proxy Details:**
    *   **Address:** `127.0.0.1`
    *   **Port:**
        *   V2rayN: `10808`
        *   Netch: `2801`
        *   SSR: `1080`
        *   Mac V2rayU: `1086`
    *   **Protocol:** Select `SOCKS5`

5.  **Enjoy!**
---
### Using System Settings (Alternative for system-wide proxy, no additional software needed):

1.  **Open OS Settings:** Access your operating system's network settings (e.g., "Network & Internet" in Windows).
2.  **Go to Proxy Settings:** Locate the proxy settings section.
3.  **Configure Proxy:**
    *   **IP:** `127.0.0.1`
    *   **Port:** `10809`
    *   **Local Host:**
        ```
        localhost;127.*;10.*;172.16.*;172.17.*;172.18.*;172.19.*;172.20.*;172.21.*;172.22.*;172.23.*;172.24.*;172.25.*;172.26.*;172.27.*;172.28.*;172.29.*;172.30.*;172.31.*;192.168.*
        ```
4.  **Enable Proxy:** Turn on the proxy setting.
5.  **V2RayN Configuration:** After setting your config in V2RayN, enable the system proxy option within the client.
6.  **Your System is Now Tunneled!**