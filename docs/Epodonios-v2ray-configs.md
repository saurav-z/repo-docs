# Free V2Ray Configs: Access the Internet Securely and Anonymously

**Get instant access to a vast library of free V2Ray configurations with frequent updates to ensure secure and private internet access.  [Check out the original repo!](https://github.com/Epodonios/v2ray-configs)**

This repository offers a regularly updated collection of V2Ray configuration files, allowing you to connect to the internet securely and anonymously using various protocols.

## Key Features:

*   **Wide Protocol Support:**  Works with Vmess, Vless, Trojan, Tuic, Shadowsocks, and ShadowsocksR.
*   **Automatic Updates:** Configs are collected and updated regularly.
*   **Multiple Format Options:** Access configurations in base64, normal, or split formats.
*   **Easy Integration:**  Compatible with popular V2Ray clients for Android (v2rayNG), iOS (Fair, Streisand), Windows, and Linux (Hiddify-next, Nekoray, v2rayN).
*   **Subscription Links:** Provides ready-to-use subscription links for easy setup and automatic updates.

## Subscription Links

Choose from various subscription links to access the latest V2Ray configurations:

*   **All Configs (Standard):** `https://github.com/Epodonios/v2ray-configs/raw/main/All_Configs_Sub.txt`
*   **All Configs (Base64):** `https://github.com/Epodonios/v2ray-configs/raw/main/All_Configs_base64_Sub.txt`

### Protocol-Specific Configs:

*   **Vless:** `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/vless.txt`
*   **Vmess:** `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/vmess.txt`
*   **Shadowsocks (ss):** `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/ss.txt`
*   **ShadowsocksR (ssr):** `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/ssr.txt`
*   **Trojan:** `https://github.com/Epodonios/v2ray-configs/raw/main/Splitted-By-Protocol/trojan.txt`

### Configs in groups of 250:

Config List 1: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub1.txt`
Config List 2: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub2.txt`
Config List 3: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub3.txt`
Config List 4: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub4.txt`
Config List 5: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub5.txt`
Config List 6: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub6.txt`
Config List 7: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub7.txt`
Config List 8: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub8.txt`
Config List 9: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub9.txt`
Config List 10: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub10.txt`
Config List 11: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub11.txt`
Config List 12: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub12.txt`
Config List 13: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub13.txt`
Config List 14: `https://raw.githubusercontent.com/Epodonios/v2ray-configs/refs/heads/main/Sub14.txt`

## How to Use:

1.  **Copy the link** of your preferred subscription type.
2.  **Paste the link** into the subscription settings of your chosen V2Ray client.
3.  **Update your subscription** periodically within your client to receive the latest configurations.

***

### **V2Ray Config Scanner (Additional Tool)**

This repository also includes a lightweight Python script to scan and test V2Ray config links.

## V2Ray Config Scanner Features

*   Supports `vmess`, vless, and other V2Ray protocols
*   Measures latency (ping) for each config
*   Sorts or filters results based on protocol and responsiveness
*   Simple, fast, and dependency-free (only requires Python)

## V2Ray Config Scanner Usage

1.  Make sure Python 3 is installed on your system.
2.  Download the sub*.txt files from this repository (they contain lists of V2Ray subscription links).
3.  Run the script and provide the path to one or more sub*.txt files as arguments.
4.  The script will start scanning and show you the protocol and ping for each config.

**Sample Output**
```
[vmess] node1.example.com - 42 ms
[vless] node2.example.net - timeout
[shadowsocks] fastnode.org - 35 ms
```

## Tunnel Entire System with Proxifier:

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

## Tunnel entire system with OS settings:

### Usage Instructions:

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

***