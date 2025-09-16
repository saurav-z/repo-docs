# Free V2Ray Configs - Access the Internet Securely and Anonymously

Looking for secure and anonymous internet access? This repository, [V2Ray Configs](https://github.com/Epodonios/v2ray-configs), provides a constantly updated collection of free V2Ray configurations.

[![Last Commit](https://img.shields.io/github/last-commit/Epodonios/v2ray-configs.svg)](https://github.com/Epodonios/v2ray-configs/commits/main)
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Update Configs](https://github.com/Epodonios/V2ray-Configs/actions/workflows/main.yml/badge.svg)](https://github.com/Epodonios/V2ray-Configs/actions/workflows/main.yml)
[![Repo Size](https://img.shields.io/github/repo-size/Epodonios/V2ray-Configs)](https://github.com/Epodonios/v2ray-configs)

Join our Telegram group for support and updates: [![Telegram](https://cdn-icons-png.flaticon.com/512/2111/2111646.png)](https://t.me/+IOG0nSifAV03ZmY0)

## Key Features:

*   **Regularly Updated:** Configurations are collected and updated frequently (every five minutes).
*   **Multiple Protocols:** Supports Vmess, Vless, Trojan, Tuic, Shadowsocks, and ShadowsocksR.
*   **Multiple Formats:** Provides configs in base64, standard, and split formats for flexibility.
*   **Wide Client Compatibility:** Works with popular V2Ray clients on Android, iOS, Windows, and Linux.
*   **Easy to Use:** Subscription links make it simple to update your configurations.

## Supported V2Ray Clients

*   **Android:** v2rayNG
*   **iOS:** fair, streisand
*   **Windows/Linux:** hiddify-next, nekoray, v2rayn

## Subscription Links:

These links provide access to the latest V2Ray configurations. Simply copy and paste these into your V2Ray client's subscription settings.

**All Configurations:**

*   [All Configurations (Standard):](https://github.com/Epodonios/v2ray-configs/raw/main/All_Configs_Sub.txt)
*   [All Configurations (Base64):](https://github.com/Epodonios/v2ray-configs/raw/main/All_Configs_base64_Sub.txt)

**Configurations by Protocol:**

*   [Vless:](https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/vless.txt)
*   [Vmess:](https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/vmess.txt)
*   [Shadowsocks (ss):](https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/ss.txt)
*   [ShadowsocksR (ssr):](https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/ssr.txt)
*   [Trojan:](https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/trojan.txt)

**Configurations Split into Lists:**

*   [Config List 1:](https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub1.txt)
*   [Config List 2:](https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub2.txt)
*   [Config List 3:](https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub3.txt)
*   [Config List 4:](https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub4.txt)
*   [Config List 5:](https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub5.txt)
*   [Config List 6:](https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub6.txt)
*   [Config List 7:](https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub7.txt)
*   [Config List 8:](https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub8.txt)
*   [Config List 9:](https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub9.txt)
*   [Config List 10:](https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub10.txt)
*   [Config List 11:](https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub11.txt)
*   [Config List 12:](https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub12.txt)
*   [Config List 13:](https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub13.txt)
*   [Config List 14:](https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub14.txt)

## How to Use:

1.  **Copy:** Get the subscription links from above.
2.  **Paste:** Add these links into the subscription settings of your V2Ray client.
3.  **Update:** Regularly use the update function within your client to ensure you have the latest configurations.

Enjoy a more secure and private internet experience!

---
---

## V2Ray Config Scanner

This lightweight Python script scans and pings a list of V2Ray configuration links, outputting their protocol and latency, allowing you to test and sort multiple V2Ray configs.

### Features

*   Supports `vmess`, `vless`, and other V2Ray protocols
*   Measures latency (ping) for each config
*   Sorts or filters results based on protocol and responsiveness
*   Simple, fast, and dependency-free (only requires Python)

### Requirements

*   Python 3.x (no external packages required)

### Usage

1.  Ensure Python 3 is installed on your system.
2.  Download the sub*.txt files from this repository.
3.  Run the script and provide the path to one or more sub*.txt files as arguments.
4.  The script will scan and show the protocol and ping for each config.

### Sample Output

```
[vmess] node1.example.com - 42 ms
[vless] node2.example.net - timeout
[shadowsocks] fastnode.org - 35 ms
```

---

## Tunneling Your Entire System (Using Proxifier):

To route all your system's traffic through a proxy, you can use a program like Proxifier.

### Instructions:

1.  **Install Proxifier:** Download and install Proxifier from https://proxifier.com/download/.
2.  **Activate Proxifier:** Use one of the following activation keys:

    *   **Portable Edition:** L6Z8A-XY2J4-BTZ3P-ZZ7DF-A2Q9C
    *   **Standard Edition:** 5EZ8G-C3WL5-B56YG-SCXM9-6QZAP
    *   **Mac OS:** P427L-9Y552-5433E-8DSR3-58Z68
3.  **Add Proxy Server:** Go to "Profile" -> "Proxy Servers" -> "Add".
4.  **Enter Proxy Information:**

    *   **Address:** 127.0.0.1
    *   **Port:** (Based on your client)
        *   V2rayN: 10808
        *   Netch: 2801
        *   SSR: 1080
        *   Mac V2rayU: 1086
    *   **Protocol:** Select SOCKS5
5.  **Enjoy!**

   Some applications might not fully tunnel. This method can resolve this.

---

## Tunneling Your Entire System (Using System Proxy - Alternative Method):

You can configure your system's built-in proxy settings to achieve a similar result.

### Instructions:

1.  **Open OS Settings:** Access your operating system's settings.
2.  **Go to Proxy Section:** Navigate to the proxy settings section.
3.  **Configure Proxy:**

    *   **IP:** 127.0.0.1
    *   **Port:** 10809
    *   **Bypass Local Addresses:**  `localhost;127.*;10.*;172.16.*;172.17.*;172.18.*;172.19.*;172.20.*;172.21.*;172.22.*;172.23.*;172.24.*;172.25.*;172.26.*;172.27.*;172.28.*;172.29.*;172.30.*;172.31.*;192.168.*`
4.  **Enable Proxy:** Turn the proxy setting to "ON."
5.  **Configure V2Ray Client:** After setting your config, turn the "Set System Proxy" option in your V2Ray client to ON.
6.  **Your System is Now Tunneled.**

```
Your friend, EPODONIOS