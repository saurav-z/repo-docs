# Free V2Ray Configs: Secure & Anonymous Internet Access

**Looking for secure and anonymous internet access?** This repository provides a constantly updated collection of free V2Ray configurations, empowering you with the tools to bypass restrictions and protect your online privacy.  [View the original repository](https://github.com/Epodonios/v2ray-configs).

[![Last Commit](https://img.shields.io/github/last-commit/barry-far/V2ray-Configs.svg)](https://github.com/Epodonios/v2ray-configs)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Update Configs](https://github.com/barry-far/V2ray-Configs/actions/workflows/main.yml/badge.svg)](https://github.com/Epodonios/V2ray-Configs/actions/workflows/main.yml)
[![Repo Size](https://img.shields.io/github/repo-size/Epodonios/V2ray-Configs)](https://github.com/Epodonios/v2ray-configs)

[Contact us on Telegram](https://t.me/+IOG0nSifAV03ZmY0)

## Key Features

*   **Regularly Updated Configurations:** Benefit from a continuously refreshed list of V2Ray configurations, ensuring you always have access to working servers.
*   **Multiple Protocol Support:**  Compatible with popular V2Ray protocols, including Vmess, Vless, Trojan, Tuic, Shadowsocks, and ShadowsocksR.
*   **Subscription Links:**  Easily integrate configurations into your V2Ray client using provided subscription links.
*   **Base64 and Split Formats:** Configurations are available in various formats to suit your specific needs and client compatibility.
*   **Cross-Platform Compatibility:** Supports popular V2Ray clients for Android, iOS, Windows, and Linux.
*   **Proxy System Usage:** provides a step by step explanation of how to tunnel your entire system using Proxy Programs
*   **System Proxy Setup:** Provides instructions to setup your proxy at the OS level.

## How to Use

1.  **Choose a V2Ray Client:** Select a compatible V2Ray client for your operating system (see client recommendations below).
2.  **Copy a Subscription Link:** Choose a subscription link from the list below.  The "All Collected Configs" link is recommended for the most comprehensive set.
3.  **Add Subscription to Client:**  Within your chosen V2Ray client, paste the subscription link into the subscription settings and save.
4.  **Update Configurations:**  Use the subscription update feature within your client to refresh the configurations regularly.

## Subscription Links

### All Configs

```
https://github.com/Epodonios/v2ray-configs/raw/main/All_Configs_Sub.txt
```

### Base64 Configs (if the above doesn't work)

```
https://github.com/Epodonios/v2ray-configs/raw/main/All_Configs_base64_Sub.txt
```

### Split by Protocol

*   **Vless:** `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/vless.txt`
*   **Vmess:** `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/vmess.txt`
*   **Shadowsocks (ss):** `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/ss.txt`
*   **ShadowsocksR (ssr):** `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/ssr.txt`
*   **Trojan:** `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/trojan.txt`

### Config Lists (Split into 250 Configs)

```
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub1.txt
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub2.txt
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub3.txt
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub4.txt
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub5.txt
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub6.txt
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub7.txt
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub8.txt
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub9.txt
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub10.txt
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub11.txt
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub12.txt
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub13.txt
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub14.txt
```

## Recommended V2Ray Clients:

*   **Android:** v2rayNG
*   **iOS:** Fair, Streisand
*   **Windows/Linux:** Hiddify-next, Nekoray, v2rayN

## Advanced: Tunneling Your Entire System

To tunnel your entire system's traffic, follow these instructions:

### Method 1: Using a Proxy Program (Proxifier)

1.  **Install Proxifier:** Download and install Proxifier from [https://proxifier.com/download/](https://proxifier.com/download/)
2.  **Activate Proxifier:**  Use the activation keys to activate the program.
    *   **Portable Edition:** `L6Z8A-XY2J4-BTZ3P-ZZ7DF-A2Q9C`
    *   **Standard Edition:** `5EZ8G-C3WL5-B56YG-SCXM9-6QZAP`
    *   **Mac OS:** `P427L-9Y552-5433E-8DSR3-58Z68`
3.  **Add a Proxy Server:**  In Proxifier, go to "Profile" > "Proxy Servers" > "Add".
4.  **Configure Proxy Server:**
    *   **Address:** `127.0.0.1`
    *   **Port:**
        *   V2rayN: `10808`
        *   Netch: `2801`
        *   SSR: `1080`
        *   Mac V2rayU: `1086`
    *   **Protocol:** Select `SOCKS5`.
5.  **Enjoy!** Your entire system traffic should now be tunneled through the V2Ray configurations.

### Method 2: System Proxy Setup (Without Proxifier)

1.  **Open OS Settings:** Access your operating system's settings.
2.  **Go to Proxy Settings:** Navigate to the proxy settings section.
3.  **Configure Proxy:**
    *   **IP:** `127.0.0.1`
    *   **Port:** `10809`
    *   **Local Host:**
    ```
    localhost;127.*;10.*;172.16.*;172.17.*;172.18.*;172.19.*;172.20.*;172.21.*;172.22.*;172.23.*;172.24.*;172.25.*;172.26.*;172.27.*;172.28.*;172.29.*;172.30.*;172.31.*;192.168.*
    ```
4.  **Enable Proxy:** Turn on the proxy setting.
5.  **V2RayN Configuration:** Return to V2RayN and configure your settings.
6.  **Set System Proxy:** Within V2RayN, enable the option to set the system proxy.
7.  **Tunneling:** Your entire system traffic should now be tunneled.

**Note:** *Some applications might require additional configuration to be fully tunneled.*

***

**Disclaimer:** *Use these configurations responsibly and in accordance with your local laws and regulations.*

***