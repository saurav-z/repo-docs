# Accelerate Your Cloudflare CDN with Optimized IPs

**Tired of slow website loading speeds?** This project helps you find and utilize the fastest Cloudflare CDN IPs for optimal performance.  Get started today by checking out the original project on GitHub: [ZhiXuanWang/cf-speed-dns](https://github.com/ZhiXuanWang/cf-speed-dns).

## Key Features:

*   **Real-time Optimized IP Lists:** Access a continuously updated list of the best-performing Cloudflare IPs.
*   **Top IP Interface:** Quickly retrieve the top-ranked Cloudflare IPs for immediate use.
    *   Top Interface: [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html)
    *   Top 10 Interface: [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **Automated DNS Updates:** Integrate with DNSPOD or DNSCF to automatically update your DNS records with the best IPs.
    *   **DNSPOD Integration:** Configure your GitHub Actions with `DOMAIN`, `SUB_DOMAIN`, `SECRETID`, `SECRETKEY`, and optionally `PUSHPLUS_TOKEN`.
    *   **DNSCF Integration:**  Set up your GitHub Actions with `CF_API_TOKEN`, `CF_ZONE_ID`, `CF_DNS_NAME`, and optionally `PUSHPLUS_TOKEN`.
*   **Push Notification Integration:** Receive notifications via PUSHPLUS to stay informed about updates.  [https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html)

## API Access

Get the top Cloudflare IPs with a simple API call:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

**Example Response:**

```
104.16.204.6,104.18.103.125
```

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Support Open Source!

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")