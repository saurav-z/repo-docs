# Optimize Your Cloudflare CDN Performance with cf-speed-dns

**Tired of slow website loading times?** This project, [cf-speed-dns](https://github.com/ZhiXuanWang/cf-speed-dns), helps you automatically find and utilize the fastest Cloudflare CDN IP addresses for optimal speed and performance.

## Key Features:

*   **Real-time Cloudflare IP Optimization:** Automatically identifies and updates the best-performing Cloudflare IP addresses every 5 minutes.
*   **Optimized IP Lists:** Access pre-sorted lists of the fastest IPs via readily available interfaces:
    *   Main List: [https://ip.164746.xyz](https://ip.164746.xyz)
    *   Top IPs (Default): [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html)
    *   Top 10 IPs: [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **Automated DNS Updates:** Integrates with DNSPOD and DNSCF for automatic domain resolution updates using the best Cloudflare IPs (fork the project and configure via GitHub Actions).
    *   **DNSPOD Integration:** Configure with `DOMAIN`, `SUB_DOMAIN`, `SECRETID`, `SECRETKEY`, and (optional) `PUSHPLUS_TOKEN`.
    *   **DNSCF Integration:** Configure with `CF_API_TOKEN`, `CF_ZONE_ID`, `CF_DNS_NAME`, and (optional) `PUSHPLUS_TOKEN`.
*   **Push Notification Integration:** Receive notifications about updates and status via PUSHPLUS ([https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html)).

## API Endpoint:

Access the top Cloudflare IPs with a simple API call:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

## API Response Example:

```
104.16.204.6,104.18.103.125
```

## Acknowledgements:

This project leverages the work of:

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Support and Infrastructure:

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")