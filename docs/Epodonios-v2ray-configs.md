# Free V2Ray Configs: Secure and Anonymous Internet Access

This repository offers a vast collection of free V2Ray configurations, updated every five minutes, to help you browse the internet securely and anonymously. Get started by visiting the [original repository](https://github.com/Epodonios/v2ray-configs) for the latest updates.

## Key Features:

*   **Regularly Updated:** Configurations are collected and updated every 5 minutes, ensuring you have access to fresh and working servers.
*   **Multiple Protocols Supported:** Supports popular V2Ray protocols including Vmess, Vless, Trojan, Tuic, Shadowsocks, and ShadowsocksR.
*   **Wide Compatibility:** Compatible with various V2Ray clients on Android, iOS, Windows, and Linux.
*   **Multiple Format Options:** Configurations available in base64, normal, and split formats for flexibility.
*   **Easy to Use:** Simply copy and paste the subscription links into your V2Ray client.

## Getting Started:

### V2Ray Clients:

Choose your preferred client based on your operating system:

*   **Android:** v2rayNG
*   **iOS:** fair, Streisand
*   **Windows/Linux:** hiddify-next, nekoray, v2rayN

### Subscription Links:

Use these links within your V2Ray client to automatically update your configurations:

*   **All Configs:**
    ```
    https://github.com/Epodonios/v2ray-configs/raw/main/All_Configs_Sub.txt
    ```
*   **Base64 Encoded (if the above doesn't work):**
    ```
    https://github.com/Epodonios/v2ray-configs/raw/main/All_Configs_base64_Sub.txt
    ```

### Split by Protocol:

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

### Split into smaller config lists:

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

### Usage Instructions:

1.  Copy a subscription link above and paste it into your V2Ray client's subscription settings.
2.  Enable the subscription update function in your client to keep your configs fresh.

### Contact:

[Telegram Contact](https://t.me/+IOG0nSifAV03ZmY0)

---

## V2Ray Config Scanner

A lightweight Python script to scan, test, and sort V2Ray configurations based on protocol and latency.

### Features:

*   Supports vmess, vless, and other V2Ray protocols
*   Measures latency (ping)
*   Sorts results based on protocol and responsiveness
*   Dependency-free (only Python required)

### Requirements:

*   Python 3.x

### How to Use:

1.  Ensure Python 3 is installed.
2.  Download the sub*.txt files from this repository.
3.  Run the script with the file paths as arguments.
4.  View protocol and ping results.

### Sample Output:

```
[vmess] node1.example.com - 42 ms
[vless] node2.example.net - timeout
[shadowsocks] fastnode.org - 35 ms
```

---

## Tunneling Your Entire System

This section provides instructions for setting up system-wide proxying for Windows and MacOS using Proxifier or directly in your system settings.

### Method 1: Using Proxifier (Recommended)

1.  Install Proxifier: [https://proxifier.com/download/](https://proxifier.com/download/)
2.  Activate Proxifier (Using the license keys provided in the original readme).
3.  Go to Profile > Proxy Servers > Add
4.  Enter the following information:
    *   IP: 127.0.0.1
    *   Port: (as per client)
        *   V2rayN: 10808
        *   Netch: 2801
        *   SSR: 1080
        *   Mac V2rayU: 1086
    *   Protocol: SOCKS5
5.  Enjoy!

### Method 2: System-Wide Proxy Setup (Without Proxifier)

1.  Open your OS settings.
2.  Go to the "Proxy" section.
3.  Set the following values:
    *   IP: 127.0.0.1
    *   Port: 10809 (or the port your client is configured to listen on)
    *   Local Host:
        ```
        localhost;127.*;10.*;172.16.*;172.17.*;172.18.*;172.19.*;172.20.*;172.21.*;172.22.*;172.23.*;172.24.*;172.25.*;172.26.*;172.27.*;172.28.*;172.29.*;172.30.*;172.31.*;192.168.*
        ```
4.  Enable the proxy.
5.  Configure your V2Ray client to use the system proxy.
6.  Your system is now tunneled entirely.

**Thank you to EPODONIOS for providing these configurations!**