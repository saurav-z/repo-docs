# Find the Fastest Cloudflare IPs with cf-speed-dns

**Optimize your Cloudflare CDN performance by automatically identifying and using the fastest IP addresses.** ([Original Repository](https://github.com/ZhiXuanWang/cf-speed-dns))

## Key Features

*   **Cloudflare IP Selection:**  Finds and provides a list of optimized Cloudflare IP addresses for improved speed and reduced latency using [CloudflareSpeedTest](https://github.com/XIU2/CloudflareSpeedTest).  Updated every 5 minutes.
*   **Real-time IP Lists:**
    *   **Main List:** [https://ip.164746.xyz](https://ip.164746.xyz)
    *   **Top IPs:** [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) (Default)
    *   **Top 10 IPs:** [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **Automated DNS Updates:**  Integrates with DNSPOD and DNSCF to automatically update your DNS records with the fastest Cloudflare IPs.
    *   **DNSPOD Integration:** Configure your GitHub Actions with `DOMAIN`, `SUB_DOMAIN`, `SECRETID`, `SECRETKEY`, and `PUSHPLUS_TOKEN`.
    *   **DNSCF Integration:** Configure your GitHub Actions with `CF_API_TOKEN`, `CF_ZONE_ID`, `CF_DNS_NAME`, and `PUSHPLUS_TOKEN`.
*   **PUSHPLUS Notification:**  Receive notifications about IP updates via [PUSHPLUS](https://www.pushplus.plus/push1.html).

## API Endpoint

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

## API Response

```javascript
104.16.204.6,104.18.103.125
```

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

##  Sponsored by

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")