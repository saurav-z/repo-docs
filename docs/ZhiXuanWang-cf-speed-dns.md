# Find the Fastest Cloudflare IPs with cf-speed-dns

**Optimize your Cloudflare CDN performance by automatically finding and utilizing the fastest IP addresses with cf-speed-dns!**  This project leverages CloudflareSpeedTest to identify the best-performing Cloudflare IPs and integrates with popular DNS providers for seamless updates. You can find the original repository here: [https://github.com/ZhiXuanWang/cf-speed-dns](https://github.com/ZhiXuanWang/cf-speed-dns).

## Key Features:

*   **Real-time Cloudflare IP Optimization:** Automatically selects and updates the fastest Cloudflare IP addresses using CloudflareSpeedTest.
*   **Updated IP Lists:** Access continuously updated lists of optimized IPs.
    *   Main List: [https://ip.164746.xyz](https://ip.164746.xyz)
    *   Top IPs (default): [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html)
    *   Top 10 IPs: [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **DNS Integration:** Automatically updates your DNS records using:
    *   DNSPOD: Configure with DOMAIN, SUB\_DOMAIN, SECRETID, SECRETKEY, and optionally PUSHPLUS\_TOKEN.
    *   DNSCF: Configure with CF\_API\_TOKEN, CF\_ZONE\_ID, CF\_DNS\_NAME, and optionally PUSHPLUS\_TOKEN.
*   **PUSHPLUS Notifications:**  Receive notifications via PUSHPLUS: [https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html)
*   **Easy API Access:** Retrieve top IPs with a simple API call:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

## API Response Example

```json
104.16.204.6,104.18.103.125
```

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Powered by

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")