# Free V2Ray Configs: Access the Internet Securely and Anonymously 🚀

This repository provides a comprehensive collection of free V2Ray configurations, empowering you to browse the internet securely and bypass censorship. Access the original repo here: [https://github.com/Epodonios/v2ray-configs](https://github.com/Epodonios/v2ray-configs)

[![Last Commit](https://img.shields.io/github/last-commit/Epodonios/v2ray-configs.svg)](https://github.com/Epodonios/v2ray-configs/commits/main)
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Update Configs](https://github.com/Epodonios/v2ray-configs/actions/workflows/main.yml/badge.svg)](https://github.com/Epodonios/V2ray-Configs/actions/workflows/main.yml)
[![Repo Size](https://img.shields.io/github/repo-size/Epodonios/V2ray-Configs)](https://github.com/Epodonios/v2ray-configs)

[<img src="https://cdn-icons-png.flaticon.com/512/2111/2111646.png" alt="Telegram" width="500" height="500">](https://t.me/+IOG0nSifAV03ZmY0) Contact us on Telegram

## Key Features:

*   **Regularly Updated:** Thousands of V2Ray configurations are collected and updated every five minutes.
*   **Multiple Protocols:** Supports Vmess, Vless, Trojan, Tuic, Shadowsocks, and ShadowsocksR protocols.
*   **Multiple Formats:** Configurations available in base64, standard, and split formats for maximum compatibility.
*   **Cross-Platform Compatibility:** Compatible with popular V2Ray clients on Android, iOS, Windows, and Linux.
*   **Easy Integration:** Subscription links for easy import into your V2Ray client.

## Supported V2Ray Clients:

*   **Android:** v2rayNG
*   **iOS:** Fair, Streisand
*   **Windows/Linux:** Hiddify-next, Nekoray, v2rayN

## Subscription Links:

**All Configs:**

*   `https://github.com/Epodonios/v2ray-configs/raw/main/All_Configs_Sub.txt`
*   `https://github.com/Epodonios/v2ray-configs/raw/main/All_Configs_base64_Sub.txt` (if the first link fails)

**Split by Protocol:**

*   Vless: `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/vless.txt`
*   Vmess: `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/vmess.txt`
*   Shadowsocks (ss): `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/ss.txt`
*   ShadowsocksR (ssr): `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/ssr.txt`
*   Trojan: `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/trojan.txt`

**Config Lists (split into groups of ~250 configs):**

*   `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub1.txt` to `Sub14.txt`

## Usage Instructions:

1.  Copy the subscription link you prefer.
2.  Go to your V2Ray client's subscription settings and paste the link.
3.  Save the settings.
4.  Regularly use the update function in your V2Ray client to keep the configs current.

***

# V2Ray Config Scanner

## Features

*   Supports `vmess`, vless, and other V2Ray protocols
*   Measures latency (ping) for each config
*   Sorts or filters results based on protocol and responsiveness
*   Simple, fast, and dependency-free (only requires Python)

## Requirements

*   Python 3.x (no external packages required)

## Usage

 1. Make sure Python 3 is installed on your system.
 2. Download the sub*.txt files from this repository (they contain lists of V2Ray subscription links).
 3. Run the script and provide the path to one or more sub*.txt files as arguments.
 4. The script will start scanning and show you the protocol and ping for each config.

Sample Output

[vmess] node1.example.com - 42 ms
[vless] node2.example.net - timeout
[shadowsocks] fastnode.org - 35 ms

## Tunnel Entire System:

### Usage Instructions:

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


## Alternative Tunneling Method:
### instruction: 

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

ur friend,EPODONIOS