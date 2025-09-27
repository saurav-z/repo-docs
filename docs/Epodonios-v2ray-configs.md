# Free V2Ray Configs - Access the Internet Securely and Anonymously

**Get instant access to a vast, regularly updated collection of V2Ray configurations to bypass restrictions and browse securely. ([Original Repo](https://github.com/Epodonios/v2ray-configs))**

**Key Features:**

*   **Automated Updates:** Configs are collected and updated every 5 minutes.
*   **Multiple Protocols:** Supports Vmess, Vless, Trojan, Tuic, Shadowsocks, and ShadowsocksR.
*   **Flexible Formats:** Configurations available in base64, standard, and split formats.
*   **Cross-Platform Compatibility:** Works with popular V2Ray clients on Android, iOS, Windows, and Linux.
*   **Subscription Links:** Easy-to-use subscription links for automatic config updates.
*   **Protocol Specific Links:** Quickly find configs for specific protocols like Vless, Vmess, Shadowsocks, SSR, and Trojan.
*   **Config Lists:** Multiple lists of configs with lists of 250 counts.

**Supported V2Ray Clients:**

*   **Android:** v2rayNG
*   **iOS:** Fair, Streisand
*   **Windows & Linux:** Hiddify-Next, Nekoray, v2rayN

## Subscription Links

Use the following subscription links within your V2Ray client to automatically receive updated configurations:

**All Configs:**

```
https://github.com/Epodonios/v2ray-configs/raw/main/All_Configs_Sub.txt
```

**All Configs (Base64):** (If the first link fails)

```
https://github.com/Epodonios/v2ray-configs/raw/main/All_Configs_base64_Sub.txt
```

**Configs by Protocol:**

*   Vless: `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/vless.txt`
*   Vmess: `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/vmess.txt`
*   Shadowsocks (ss): `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/ss.txt`
*   ShadowsocksR (ssr): `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/ssr.txt`
*   Trojan: `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/trojan.txt`

**Config Lists (250 configs per list):**

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

## How to Use

1.  **Copy & Paste:** Copy one of the subscription links above.
2.  **Client Setup:**  Go to your V2Ray client's subscription settings and paste the link.
3.  **Update:** Regularly use the "update subscription" function in your client to refresh configs.

**Enjoy secure and private internet access!**

---

## Additional System-Wide Tunneling Options

### Using Proxifier (Windows & MacOS)

1.  **Install Proxifier:** Download and install Proxifier (license keys provided in original README).
2.  **Configure Proxy Server:**  Add a proxy server in Proxifier settings (Profile -> Proxy Servers -> Add).
    *   **Address:** 127.0.0.1
    *   **Port:**  Use the port corresponding to your V2Ray client:
        *   V2rayN: 10808
        *   Netch: 2801
        *   SSR: 1080
        *   Mac V2rayU: 1086
    *   **Protocol:** SOCKS5
3.  **Enjoy System-Wide Tunneling!**

### Using System Proxy Settings (All OS)

1.  **Open OS Settings:** Access your operating system's proxy settings.
2.  **Configure Proxy:**
    *   **IP:** 127.0.0.1
    *   **Port:**  10809 (or the port used by your V2Ray client)
    *   **Bypass:** (Optional, can be added to the local host section to bypass localhost)
    ```
    localhost;127.*;10.*;172.16.*;172.17.*;172.18.*;172.19.*;172.20.*;172.21.*;172.22.*;172.23.*;172.24.*;172.25.*;172.26.*;172.27.*;172.28.*;172.29.*;172.30.*;172.31.*;192.168.*
    ```
3.  **Enable Proxy:** Activate the proxy settings.
4.  **V2Ray Client:** Configure your V2Ray client to use the system proxy.
5.  **System Tunnelling :** Your System is Now Tunneled Entirely

---

**Note:** For more information and support, you can reach out to the owner on Telegram.

---