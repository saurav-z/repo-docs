# Free V2Ray Configs: Secure and Anonymous Internet Access

Access the internet securely and anonymously with this comprehensive collection of free V2Ray configuration files. This repository, [Epodonios/v2ray-configs](https://github.com/Epodonios/v2ray-configs), provides regularly updated V2Ray configurations for various protocols, ensuring you have access to fast and reliable connections.

[![GitHub last commit](https://img.shields.io/github/last-commit/barry-far/V2ray-Configs.svg)](https://github.com/Epodonios/v2ray-configs)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Update Configs](https://github.com/barry-far/V2ray-Configs/actions/workflows/main.yml/badge.svg)](https://github.com/Epodonios/V2ray-Configs/actions/workflows/main.yml)
![GitHub repo size](https://img.shields.io/github/repo-size/Epodonios/V2ray-Configs)

## Key Features:

*   **Regularly Updated:** Configurations are automatically collected and updated every five minutes.
*   **Multiple Protocols:** Supports Vmess, Vless, Trojan, Tuic, Shadowsocks, and ShadowsocksR protocols.
*   **Subscription Links:** Provides subscription links for easy integration with V2Ray clients.
*   **Multiple Formats:** Available in base64, normal, and split formats for flexibility.
*   **Wide Compatibility:** Works with popular V2Ray clients on Android, iOS, Windows, and Linux.

## Get Started: Subscription Links

Use these subscription links within your preferred V2Ray client to access the latest configurations.  Remember to periodically update your subscription within your client to get the newest configs.

**Main Subscription Links:**

*   All Configs:  `https://github.com/Epodonios/v2ray-configs/raw/main/All_Configs_Sub.txt`
*   Base64 Encoded Configs: `https://github.com/Epodonios/v2ray-configs/raw/main/All_Configs_base64_Sub.txt`

**Protocol-Specific Links:**

*   Vless: `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/vless.txt`
*   Vmess: `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/vmess.txt`
*   Shadowsocks (ss): `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/ss.txt`
*   ShadowsocksR (ssr): `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/ssr.txt`
*   Trojan: `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/trojan.txt`

**Config Lists (Split into 250 count chunks):**

*   Config List 1: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub1.txt`
*   Config List 2: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub2.txt`
*   Config List 3: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub3.txt`
*   Config List 4: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub4.txt`
*   Config List 5: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub5.txt`
*   Config List 6: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub6.txt`
*   Config List 7: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub7.txt`
*   Config List 8: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub8.txt`
*   Config List 9: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub9.txt`
*   Config List 10: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub10.txt`
*   Config List 11: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub11.txt`
*   Config List 12: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub12.txt`
*   Config List 13: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub13.txt`
*   Config List 14: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub14.txt`

## Compatible V2Ray Clients:

*   **Android:** v2rayNG
*   **iOS:** fair, Streisand
*   **Windows & Linux:** Hiddify-next, Nekoray, v2rayN

## Additional Resources:

*   [Contact us on Telegram](https://t.me/+IOG0nSifAV03ZmY0)

***

**(The following sections can be moved to a separate document, if needed, to keep the main README focused on config usage.)**

## V2Ray Config Scanner (for advanced users)

This repository also includes a simple Python script for scanning and testing V2Ray configurations.

### Features:

*   Supports `vmess`, `vless`, and other V2Ray protocols
*   Measures latency (ping) for each config
*   Sorts or filters results based on protocol and responsiveness
*   Simple, fast, and dependency-free (only requires Python)

### Requirements:

*   Python 3.x

### Usage:

1.  Ensure Python 3 is installed.
2.  Download the sub*.txt files (containing the configuration links) from this repository.
3.  Run the script, providing the path(s) to the `sub*.txt` files as arguments.

## System-Wide Tunneling with Proxifier

To tunnel your entire system's traffic, you can use the Proxifier program:

### Instructions:

1.  Install Proxifier:  https://proxifier.com/download/
2.  Activate the program.  (Portable Edition Key provided in the original README).
3.  Go to "Profile" > "Proxy Servers" and click "Add".
4.  Enter the following information:
    *   Address: `127.0.0.1`
    *   Port:  (Choose the port based on your V2Ray client)
        *   V2rayN: `10808`
        *   Netch: `2801`
        *   SSR: `1080`
        *   Mac V2rayU: `1086`
    *   Protocol: `SOCKS5`
5.  Enjoy!

## Alternative: System Proxy Settings

You can also configure your operating system's proxy settings.

### Instructions:

1.  Open your OS settings and go to the "Proxy" section.
2.  Enter the following values:
    *   Address: `127.0.0.1`
    *   Port: `10809` (or the port configured in your V2Ray client)
    *   Bypass for these hosts & domains:  `localhost;127.*;10.*;172.16.*;172.17.*;172.18.*;172.19.*;172.20.*;172.21.*;172.22.*;172.23.*;172.24.*;172.25.*;172.26.*;172.27.*;172.28.*;172.29.*;172.30.*;172.31.*;192.168.*`
3.  Enable the proxy setting.
4.  In V2rayN, after setting your config, enable the system proxy option.
5.  Your system's traffic is now tunneled.