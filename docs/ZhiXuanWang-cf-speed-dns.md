# Optimize Your Cloudflare CDN Performance with cf-speed-dns

**Tired of slow website speeds?** cf-speed-dns automatically finds the fastest Cloudflare IP addresses for you, ensuring optimal performance. Find the original repository [here](https://github.com/ZhiXuanWang/cf-speed-dns).

## Key Features

*   **Automated IP Selection:**  Every 5 minutes, cf-speed-dns identifies and updates the best-performing Cloudflare IP addresses.
*   **Real-time IP List:** Access a constantly updated list of optimized IPs via web interfaces:
    *   [https://ip.164746.xyz](https://ip.164746.xyz) (Main List)
    *   [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) (Top IPs - Default)
    *   [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html) (Top 10 IPs)
*   **DNSPOD Integration:**  Seamlessly integrates with DNSPOD for automated DNS updates (Fork the project and configure your secrets).
    *   Configure using `DOMAIN`, `SUB_DOMAIN`, `SECRETID`, `SECRETKEY`, and optionally `PUSHPLUS_TOKEN`.
*   **DNSCF Integration:** Integrates with Cloudflare for automated DNS updates (Fork the project and configure your secrets).
    *   Configure using `CF_API_TOKEN`, `CF_ZONE_ID`, `CF_DNS_NAME`, and optionally `PUSHPLUS_TOKEN`.
*   **PUSHPLUS Notifications:** Receive instant notifications via PUSHPLUS for updates and alerts. ([https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html))

## API Access

Access the top-performing IP addresses using the following API endpoint:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

## API Response Example

The API returns a comma-separated list of the fastest Cloudflare IP addresses:

```
104.16.204.6,104.18.103.125
```

## Acknowledgements

Special thanks to:

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Advertisement

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")