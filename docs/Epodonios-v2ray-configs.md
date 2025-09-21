# Free V2Ray Configs: Access the Internet Securely and Anonymously

This repository offers a constantly updated collection of free V2Ray configurations for secure and private internet access. ([Original Repo](https://github.com/Epodonios/v2ray-configs))

[![Last Commit](https://img.shields.io/github/last-commit/barry-far/V2ray-Configs.svg)](https://github.com/Epodonios/v2ray-configs)
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Update Configs](https://github.com/barry-far/V2ray-Configs/actions/workflows/main.yml/badge.svg)](https://github.com/Epodonios/V2ray-Configs/actions/workflows/main.yml)
[![Repo Size](https://img.shields.io/github/repo-size/Epodonios/V2ray-Configs)](https://github.com/Epodonios/v2ray-configs)

[Contact us on Telegram](https://t.me/+IOG0nSifAV03ZmY0)

## Key Features:

*   **Always Updated:** Configurations are collected and updated every five minutes.
*   **Multiple Protocols:** Supports Vmess, Vless, Trojan, Tuic, Shadowsocks, and ShadowsocksR.
*   **Multiple Formats:** Available in base64, standard, and protocol-split formats.
*   **Subscription Links:** Easy-to-use subscription links for easy configuration in your V2Ray client.
*   **Cross-Platform Compatibility:** Works with popular V2Ray clients on Android, iOS, Windows, and Linux.
*   **Configuration splitting:** configs are split in 250 count for better usage .
*   **System tunnel options**: you can tunnel entire system with proxy program and system setting.

## Subscription Links

Access the latest configurations through these subscription links:

*   **All Configurations:**

```
https://github.com/Epodonios/v2ray-configs/raw/main/All_Configs_Sub.txt
```

*   **Base64 Encoded (if the above fails):**

```
https://github.com/Epodonios/v2ray-configs/raw/main/All_Configs_base64_Sub.txt
```

*   **Split by Protocol:**
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

*   **Split into 250 Config Lists:**

    *   Config List 1:
        ```
        https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub1.txt
        ```
    *   Config List 2:
        ```
        https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub2.txt
        ```
    *   Config List 3:
        ```
        https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub3.txt
        ```
    *   Config List 4:
        ```
        https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub4.txt
        ```
    *   Config List 5:
        ```
        https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub5.txt
        ```
    *   Config List 6:
        ```
        https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub6.txt
        ```
    *   Config List 7:
        ```
        https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub7.txt
        ```
    *   Config List 8:
        ```
        https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub8.txt
        ```
    *   Config List 9:
        ```
        https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub9.txt
        ```
    *   Config List 10:
        ```
        https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub10.txt
        ```
    *   Config List 11:
        ```
        https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub11.txt
        ```
    *   Config List 12:
        ```
        https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub12.txt
        ```
    *   Config List 13:
        ```
        https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub13.txt
        ```
    *   Config List 14:
        ```
        https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub14.txt
        ```

## How to Use:

1.  **Copy & Paste:** Copy the provided subscription links.
2.  **Configure Your Client:** Open your V2Ray client and paste the link in the subscription settings.
3.  **Update Regularly:** Use the update function in your V2Ray client to keep your configurations up-to-date.

## System-Wide Tunneling

You can tunnel the entire system with a proxy program or system settings, these are the methods:

### Method 1: Using a Proxy Program (e.g., Proxifier)

1.  **Install Proxifier:** Download and install the Proxifier program ([https://proxifier.com/download/](https://proxifier.com/download/)).
2.  **Activate Proxifier:** Use one of the activation keys provided.
3.  **Add Proxy Server:** In Proxifier, go to "Profile" > "Proxy Servers" and click "Add."
4.  **Enter Proxy Details:**
    *   Address: `127.0.0.1`
    *   Port:
        *   V2rayN: `10808`
        *   Netch: `2801`
        *   SSR: `1080`
        *   Mac V2rayU: `1086`
    *   Protocol: `SOCKS5`
5.  **Enjoy!**

### Method 2: Using System Settings

1.  **Open System Settings:** Go to your operating system's proxy settings.
2.  **Configure Proxy:**
    *   Address: `127.0.0.1`
    *   Port: `10809`
    *   Bypass local addresses:
        ```
        localhost;127.*;10.*;172.16.*;172.17.*;172.18.*;172.19.*;172.20.*;172.21.*;172.22.*;172.23.*;172.24.*;172.25.*;172.26.*;172.27.*;172.28.*;172.29.*;172.30.*;172.31.*;192.168.*
        ```
3.  **Enable Proxy:** Turn the proxy setting ON.
4.  **V2Ray Client:** In your V2Ray client, set the system proxy to ON after setting your configuration.

## V2Ray Config Scanner

A simple Python script is included to test and sort the V2Ray config links in order to find the best options.

## Requirements

- Python 3.x (no external packages required)

## Usage

1. Make sure Python 3 is installed on your system.
2. Download the sub\*.txt files from this repository (they contain lists of V2Ray subscription links).
3. Run the script and provide the path to one or more sub\*.txt files as arguments.
4. The script will start scanning and show you the protocol and ping for each config.

Sample Output

```bash
[vmess] node1.example.com - 42 ms
[vless] node2.example.net - timeout
[shadowsocks] fastnode.org - 35 ms
```

**Enjoy secure and anonymous internet access!**