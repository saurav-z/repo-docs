## Speed Up Your Cloudflare CDN with cf-speed-dns

**cf-speed-dns** helps you find the fastest Cloudflare IP addresses and automatically updates your DNS records for optimal performance.  (See the original repository [here](https://github.com/ZhiXuanWang/cf-speed-dns).)

### Key Features:

*   **Cloudflare IP Optimization:** Identifies and provides a list of the fastest Cloudflare IPs using CloudflareSpeedTest.
*   **Real-time IP Updates:** Access the latest optimized IPs via a dedicated webpage: [https://ip.164746.xyz](https://ip.164746.xyz)
*   **Top IP Interfaces:** Get the top-performing IPs via readily available interfaces:
    *   Top Interface (Default): [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html)
    *   Top 10 Interface: [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **Automated DNS Updates:**  Supports automated DNS record updates using DNSPOD and DNSCF. Configuration via GitHub Actions.
    *   **DNSPOD Integration:** Requires DOMAIN, SUB\_DOMAIN, SECRETID, SECRETKEY, and optionally PUSHPLUS\_TOKEN for notifications.
    *   **DNSCF Integration:** Requires CF\_API\_TOKEN, CF\_ZONE\_ID, CF\_DNS\_NAME, and optionally PUSHPLUS\_TOKEN for notifications.
*   **PUSHPLUS Notifications:**  Receive notifications about updates using PUSHPLUS:  [https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html)

### Example API Request

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

### Example API Response

```
104.16.204.6,104.18.103.125
```

### Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

### Powered by

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")