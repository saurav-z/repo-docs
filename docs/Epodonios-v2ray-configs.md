# Enhance Your Internet Freedom with Free V2Ray Configurations

This repository, a robust collection of free V2Ray configurations, offers a secure and private way to access the internet, updated regularly. ([Original Repo](https://github.com/Epodonios/v2ray-configs))

## Key Features:

*   **Automatic Updates:** Configurations are collected and updated every five minutes.
*   **Multiple Protocols:** Supports Vmess, Vless, Trojan, Tuic, Shadowsocks, and ShadowsocksR.
*   **Subscription Links:** Easy-to-use subscription links for seamless configuration updates.
*   **Cross-Platform Compatibility:** Compatible with V2Ray clients on Android, iOS, Windows, and Linux.
*   **Multiple Formats:** Configs available in base64, normal, and split formats.
*   **Easy Setup:** Simple instructions for setting up configurations on various devices.

## Accessing Configurations:

Choose your preferred subscription link below to get started:

*   **All Collected Configs:** `https://github.com/Epodonios/v2ray-configs/raw/main/All_Configs_Sub.txt`
*   **Base64 Encoded Configs (If the above doesn't work):** `https://github.com/Epodonios/v2ray-configs/raw/main/All_Configs_base64_Sub.txt`

### Split by Protocol:

*   **Vless:** `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/vless.txt`
*   **Vmess:** `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/vmess.txt`
*   **Shadowsocks (ss):** `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/ss.txt`
*   **ShadowsocksR (ssr):** `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/ssr.txt`
*   **Trojan:** `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/trojan.txt`

### Config Lists (Split into sets of ~250 configurations):

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

## How to Use:

1.  **Copy the Link:** Choose your desired subscription link.
2.  **Paste into Client:** In your V2Ray client (e.g., v2rayNG, Fair, etc.), go to the subscription settings and paste the link.
3.  **Update Regularly:** Use the update function within your V2Ray client to stay current with fresh configurations.

## Disclaimer

This project provides configurations that are collected from public sources. The maintainers are not responsible for any misuse of the configurations. Users are encouraged to use the configurations responsibly and in accordance with their local laws and regulations.

## V2Ray Config Scanner (For Testing & Performance)

A Python script to scan V2Ray configuration links, measure latency, and output protocol and ping.

### Features:

*   Supports various V2Ray protocols (vmess, vless, etc.).
*   Measures latency for each configuration.
*   Sorts or filters results based on protocol and performance.
*   Simple, fast, and Python-based (no external packages required).

### Requirements:

*   Python 3.x

### Usage:

1.  Ensure Python 3 is installed.
2.  Download `sub*.txt` files from this repository.
3.  Run the script, providing the path to the `sub*.txt` files.

## Tunnelling your entire system:

You can configure your system to use a proxy program such as Proxifier or use the OS's built in proxy settings.

### **With Proxy Program (Example: Proxifier)**

1.  Install Proxifier:  https://proxifier.com/download/
2.  Activate the program (Activation keys are provided in the original readme).
3.  Add a Proxy Server:
    *   Go to "Profile" -> "Proxy Servers" -> "Add."
    *   IP: 127.0.0.1
    *   Port: (Varies based on your setup and client. Examples given include V2rayN: 10808, Netch: 2801, SSR: 1080, Mac V2rayU: 1086)
    *   Protocol: SOCKS5
4.  Enjoy!

### **Without Proxy Program (Using OS Settings):**

1.  Open your OS Settings -> Proxy Section.
2.  Configure the Proxy:
    *   IP: 127.0.0.1
    *   Port: 10809 (or the port your V2Ray client is configured on)
    *   Local host:  `localhost;127.*;10.*;172.16.*;172.17.*;172.18.*;172.19.*;172.20.*;172.21.*;172.22.*;172.23.*;172.24.*;172.25.*;172.26.*;172.27.*;172.28.*;172.29.*;172.30.*;172.31.*;192.168.*`
3.  Activate the proxy.
4.  In your V2Ray client, enable "Set System Proxy".

Your friend, EPODONIOS