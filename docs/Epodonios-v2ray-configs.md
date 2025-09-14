# Secure Your Internet Access with Free V2Ray Configs

Access the internet securely and bypass restrictions with [Epodonios' V2Ray Configs](https://github.com/Epodonios/v2ray-configs), your go-to source for free and frequently updated V2Ray configurations.

[![GitHub last commit](https://img.shields.io/github/last-commit/barry-far/V2ray-Configs.svg)](https://github.com/Epodonios/V2ray-Configs) [![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)  [![Update Configs](https://github.com/barry-far/V2ray-Configs/actions/workflows/main.yml/badge.svg)](https://github.com/Epodonios/V2ray-Configs/actions/workflows/main.yml) ![GitHub repo size](https://img.shields.io/github/repo-size/Epodonios/V2ray-Configs)  

<a href="https://t.me/+IOG0nSifAV03ZmY0" target="_blank">
  <img src="https://cdn-icons-png.flaticon.com/512/2111/2111646.png" alt="Telegram" width="500" height="500"> Contact Us on Telegram
</a>

## Key Features

*   **Regularly Updated:** Configurations are gathered and updated every 5 minutes, ensuring you always have access to working configs.
*   **Multiple Protocol Support:** Supports popular protocols like Vmess, Vless, Trojan, Tuic, Shadowsocks, and ShadowsocksR.
*   **Subscription Links:** Easily integrate the configs into your preferred V2Ray clients with provided subscription links.
*   **Base64 and Split Formats:** Access configs in base64, standard, or split formats for enhanced compatibility.
*   **Cross-Platform Compatibility:** Works with various V2Ray client applications across Android, iOS, Windows, and Linux.

## Supported V2Ray Clients

*   **Android:** v2rayNG
*   **iOS:** Fair, Streisand
*   **Windows & Linux:** Hiddify-Next, Nekoray, v2rayN

## Subscription Links - Get Started Now!

Choose the format that works best for your V2Ray client.  Copy and paste these subscription links into your client's settings, and then use the update function in your client to update the configs:

**All Configs (Recommended):**

```
https://github.com/Epodonios/v2ray-configs/raw/main/All_Configs_Sub.txt
```

**All Configs (Base64):** (If the above link doesn't work)

```
https://github.com/Epodonios/v2ray-configs/raw/main/All_Configs_base64_Sub.txt
```

**Configs by Protocol:**

*   Vless:
    ```
    https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/vless.txt
    ```
*   Vmess:
    ```
    https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/vmess.txt
    ```
*   Shadowsocks (ss):
    ```
    https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/ss.txt
    ```
*   ShadowsocksR (ssr):
    ```
    https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/ssr.txt
    ```
*   Trojan:
    ```
    https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/trojan.txt
    ```

**Configs Split into Lists (250 configs per list):**

These lists provide another way to obtain configs, in case your client has issues handling the single large files.
```
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub1.txt
```
```
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub2.txt
```
```
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub3.txt
```
```
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub4.txt
```
```
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub5.txt
```
```
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub6.txt
```
```
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub7.txt
```
```
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub8.txt
```
```
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub9.txt
```
```
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub10.txt
```
```
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub11.txt
```
```
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub12.txt
```
```
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub13.txt
```
```
https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub14.txt
```

## Installation

1.  Copy the subscription link you prefer from the list above.
2.  In your chosen V2Ray client, go to the subscription settings and paste the link.
3.  Save the settings.
4.  Regularly update the subscription within your client to stay current with fresh configs.

Enjoy your secure and unrestricted internet access!

---

## V2Ray Config Scanner (for Developers and Advanced Users)

This repository also includes a lightweight Python script for scanning and testing V2Ray configurations.

### Features

*   Supports `vmess`, `vless`, `trojan`, `shadowsocks`, and `shadowsocksR` protocols.
*   Measures latency (ping) for each config.
*   Sorts or filters results based on protocol and responsiveness.
*   Simple, fast, and dependency-free (requires only Python).

### Requirements

*   Python 3.x (no external packages needed).

### Usage

1.  Ensure Python 3 is installed.
2.  Download one or more of the `Sub*.txt` files (containing lists of V2Ray subscription links) from this repository.
3.  Run the Python script, providing the path to the downloaded file(s) as arguments.
4.  The script will scan and display the protocol and ping results for each config.

**Example Output:**

```
[vmess] node1.example.com - 42 ms
[vless] node2.example.net - timeout
[shadowsocks] fastnode.org - 35 ms
```

---

## Tunneling Your Entire System (Advanced)

The following methods can be used to route your system's traffic through a proxy server.

### Using Proxifier:

1.  **Install Proxifier:** Download and install Proxifier from [https://proxifier.com/download/](https://proxifier.com/download/).
2.  **Activate Proxifier:** Use a license key.
    *   Portable Edition: `L6Z8A-XY2J4-BTZ3P-ZZ7DF-A2Q9C`
    *   Standard Edition: `5EZ8G-C3WL5-B56YG-SCXM9-6QZAP`
    *   Mac OS: `P427L-9Y552-5433E-8DSR3-58Z68`
3.  **Add Proxy Server:** In Proxifier, go to "Profile" -> "Proxy Servers" and click "Add".
4.  **Configure Proxy Settings:**
    *   IP: `127.0.0.1`
    *   Port:
        *   V2rayN: `10808`
        *   Netch: `2801`
        *   SSR: `1080`
        *   Mac V2rayU: `1086`
    *   Protocol: SOCKS5
5.  **Enjoy!**  Proxifier will now route all your traffic through the proxy.

### Using System Proxy Settings (Alternative):

1.  **Open OS Settings:** Access your system's proxy settings.
2.  **Configure Proxy:**
    *   IP: `127.0.0.1`
    *   Port: `10809`
    *   Local Host:
        ```
        localhost;127.*;10.*;172.16.*;172.17.*;172.18.*;172.19.*;172.20.*;172.21.*;172.22.*;172.23.*;172.24.*;172.25.*;172.26.*;172.27.*;172.28.*;172.29.*;172.30.*;172.31.*;192.168.*
        ```
3.  **Enable Proxy:** Activate the proxy settings.
4.  **V2Ray Client Configuration:** In your V2Ray client, enable system proxy settings.
5.  **Complete!**  Your system's traffic will now be tunneled.