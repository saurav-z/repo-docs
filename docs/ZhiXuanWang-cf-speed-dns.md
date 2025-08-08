# Optimize Your Cloudflare CDN with cf-speed-dns

**Instantly find and deploy the fastest Cloudflare CDN IPs with cf-speed-dns, ensuring optimal website performance.**  [View the original repository on GitHub](https://github.com/ZhiXuanWang/cf-speed-dns).

## Key Features:

*   **Automatic Cloudflare IP Optimization:** Automatically selects and updates the best-performing Cloudflare IP addresses every 5 minutes.
*   **Real-time IP List:** Access a real-time list of optimized IPs for Cloudflare CDN via a web interface: [https://ip.164746.xyz](https://ip.164746.xyz).
*   **Top IP Interfaces:** Conveniently retrieve the top-performing IPs.
    *   Top Interface (Default): [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html)
    *   Top 10 Interface: [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **DNSPOD Integration:**  Automated DNS record updates via DNSPOD. Requires forking the project and configuring your credentials (DOMAIN, SUB\_DOMAIN, SECRETID, SECRETKEY, PUSHPLUS\_TOKEN) in GitHub Actions secrets and variables.
*   **DNSCF Integration:** Automated DNS record updates via Cloudflare API. Requires forking the project and configuring your credentials (CF\_API\_TOKEN, CF\_ZONE\_ID, CF\_DNS\_NAME, PUSHPLUS\_TOKEN) in GitHub Actions secrets and variables.
*   **PUSHPLUS Notifications:** Integrated notification system to keep you informed using PUSHPLUS:  [https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html).

## API Endpoint

Retrieve the top-performing IPs using the following endpoint:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

## API Response

The API returns a comma-separated list of the top-performing IP addresses:

```
104.16.204.6,104.18.103.125
```

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Powered by

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")