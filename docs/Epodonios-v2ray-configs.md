# Secure & Anonymous Internet Access with Free V2Ray Configurations

**Need secure and anonymous internet access?** This repository provides a regularly updated collection of free V2Ray configuration files for various protocols. [(View on GitHub)](https://github.com/Epodonios/v2ray-configs)

*   **Comprehensive Configuration:** Access a wide array of V2Ray configurations, including Vmess, Vless, Trojan, Tuic, Shadowsocks, and ShadowsocksR.
*   **Automatic Updates:** Configurations are collected and updated frequently, ensuring you have access to the latest available options.
*   **Multiple Formats:** Choose from base64, regular, or split-protocol formats for compatibility with various V2Ray clients.
*   **Easy Integration:** Subscription links are provided for easy setup with popular V2Ray clients on Android, iOS, Windows, and Linux.

## Getting Started

1.  **Choose a Client:** Select a V2Ray client compatible with your operating system (see list below).
2.  **Copy Subscription Links:** Use the subscription links provided to automatically update your configurations.
3.  **Update Regularly:** Use the update function in your V2Ray client to refresh configurations.

## Subscription Links

**All Configurations:**

*   `https://github.com/Epodonios/v2ray-configs/raw/main/All_Configs_Sub.txt`
*   `https://github.com/Epodonios/v2ray-configs/raw/main/All_Configs_base64_Sub.txt` (if the first link doesn't work)

**Split by Protocol:**

*   Vless: `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/vless.txt`
*   Vmess: `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/vmess.txt`
*   Shadowsocks (ss): `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/ss.txt`
*   ShadowsocksR (ssr): `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/ssr.txt`
*   Trojan: `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/trojan.txt`

**Split into Configuration Lists (250 configs each):**

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

## Supported V2Ray Clients

*   **Android:** v2rayNG
*   **iOS:** Fair, Streisand
*   **Windows & Linux:** Hiddify-next, NekoRay, v2rayN

## System-Wide Tunneling with Proxifier

For comprehensive internet access and to proxy all system traffic, you can use Proxifier:

1.  **Install Proxifier:**  Download and install Proxifier (link provided in original README).
2.  **Activate Proxifier:** Use the activation key provided.
3.  **Configure Proxy Server:**  Add a proxy server in Proxifier settings.
    *   IP: `127.0.0.1`
    *   Port:  (See original README for port values based on your V2Ray client - typically, 10808 (V2rayN), 2801 (Netch), 1080 (SSR), or 1086 (V2rayU for Mac))
    *   Protocol: SOCKS5

## System-Wide Tunneling without Proxifier

1.  **Open OS Settings:** Go to your operating system's proxy settings.
2.  **Configure Proxy:** Set these values:
    *   IP: `127.0.0.1`
    *   Port: `10809`
    *   Bypass for local addresses:
        `localhost;127.*;10.*;172.16.*;172.17.*;172.18.*;172.19.*;172.20.*;172.21.*;172.22.*;172.23.*;172.24.*;172.25.*;172.26.*;172.27.*;172.28.*;172.29.*;172.30.*;172.31.*;192.168.*`
3.  **Enable Proxy:** Turn on the proxy setting.
4.  **V2RayN Configuration:** In V2RayN, set your configuration to use the system proxy.

---