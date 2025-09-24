# Secure & Free V2Ray Configurations for Anonymous Internet Access

**Looking for a way to browse the internet securely and privately?** This repository, ([Original Repo](https://github.com/Epodonios/v2ray-configs)), offers a constantly updated collection of free V2Ray configuration files.

## Key Features:

*   **Wide Protocol Support:** Access configurations for Vmess, Vless, Trojan, Tuic, Shadowsocks, and ShadowsocksR protocols.
*   **Automated Updates:** Configurations are collected and updated frequently.
*   **Multiple Format Options:** Receive configurations in base64, normal, or split formats to suit your needs.
*   **Cross-Platform Compatibility:** Works with popular V2Ray clients on Android, iOS, Windows, and Linux.
*   **Easy to Use:** Simply copy and paste subscription links into your V2Ray client.
*   **Various Proxy Software Solutions** Use your proxy softwares for a system wide anonymous connection

## How to Use:

1.  **Choose Your Client:** Select a V2Ray client compatible with your operating system (examples provided in original README).
2.  **Get Subscription Links:** Choose from the following subscription links to import configurations:

    *   **All Configurations:**
        ```
        https://github.com/Epodonios/v2ray-configs/raw/main/All_Configs_Sub.txt
        ```
        (If the above link doesn't work, try the base64 configurations)
        ```
        https://github.com/Epodonios/v2ray-configs/raw/main/All_Configs_base64_Sub.txt
        ```
    *   **Split by Protocol:** Access configurations by protocol for more control.
        *   Vless:
            ```
            https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/vless.txt
            ```
        *   Vmess:
            ```
            https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/vmess.txt
            ```
        *   ss:
            ```
            https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/ss.txt
            ```
        *   ssr:
            ```
            https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/ssr.txt
            ```
        *   Trojan:
            ```
            https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/trojan.txt
            ```
    *   **Split in 250 count of configs:** Access configurations by protocol for more control.
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
3.  **Import and Update:** Copy the link of your choice and paste it into your V2Ray client's subscription settings and save it. Refresh your subscription periodically within your V2Ray client to ensure you have the latest configurations.

## Additional Tools

This repository also provides a lightweight Python script:

### V2Ray Config Scanner

This lightweight Python script that scans and pings a list of V2Ray configuration links (vmess, vless, etc.), and outputs their protocol and latency. Useful for testing and sorting multiple V2Ray configs based on performance.

#### Features

- Supports `vmess`, vless, and other V2Ray protocols
- Measures latency (ping) for each config
- Sorts or filters results based on protocol and responsiveness
- Simple, fast, and dependency-free (only requires Python)

#### Requirements

- Python 3.x (no external packages required)

#### Usage

1.  Make sure Python 3 is installed on your system.
2.  Download the sub\*.txt files from this repository (they contain lists of V2Ray subscription links).
3.  Run the script and provide the path to one or more sub\*.txt files as arguments.
4.  The script will start scanning and show you the protocol and ping for each config.

Sample Output

```
[vmess] node1.example.com - 42 ms
[vless] node2.example.net - timeout
[shadowsocks] fastnode.org - 35 ms
```

## System-Wide Tunneling using Proxy software

  1-First, install the Proxifier program.

  https://proxifier.com/download/
  
  2-Activate the program:

Activation keys:

Portable Edition:  

    L6Z8A-XY2J4-BTZ3P-ZZ7DF-A2Q9C

Standard Edition: 
      
      5EZ8G-C3WL5-B56YG-SCXM9-6QZAP

Mac OS:

     P427L-9Y552-5433E-8DSR3-58Z68

3-Go to the Profile section and select the Proxy Server. In the displayed section, click on Add.

4-Enter the following information:

IP: Enter 127.0.0.1

Port: Depending on the version you are using, enter:

V2rayN: 10808

Netch: 2801

SSR: 1080

Mac V2rayU: 1086

Protocol: Select SOCKS5

5-Enjoy!

Some installed programs on the system, like Spotube, might not fully tunnel. This issue can be resolved with this method.

## System-Wide Tunneling: OS Setting Solution

  1- open your OS setting 

  2- go to proxy section

  3- in proxy section set this values : 
    ip : 127.0.0.1
  
    port : 10809
  
    local host : 
    ```
  localhost;127.*;10.*;172.16.*;172.17.*;172.18.*;172.19.*;172.20.*;172.21.*;172.22.*;172.23.*;172.24.*;172.25.*;172.26.*;172.27.*;172.28.*;172.29.*;172.30.*;172.31.*;192.168.*
  ```
   4- then set it up with ON key 

   5- back to v2rayn and after set your config turn it to set system proxy 

   6- now your system tunneled entirely