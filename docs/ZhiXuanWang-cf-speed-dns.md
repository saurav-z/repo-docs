# CF Speed DNS: Optimize Your Cloudflare Performance

**Instantly improve your website's speed and reliability by leveraging the fastest Cloudflare CDN IPs with CF Speed DNS.** This project automatically finds and updates the best Cloudflare IP addresses for optimal performance.  Check out the original repository [here](https://github.com/ZhiXuanWang/cf-speed-dns).

## Key Features

*   **Automatic IP Selection:**  Continuously identifies and updates the fastest Cloudflare CDN IPs every 5 minutes using CloudflareSpeedTest.
*   **Optimized IP Lists:**  Provides easy access to optimized IP lists, including:
    *   Real-time updated list: [https://ip.164746.xyz](https://ip.164746.xyz)
    *   Top-performing IPs (default): [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html)
    *   Top 10 IPs: [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **Dynamic DNS Integration:**  Supports real-time domain name resolution updates for DNSPOD and DNSCF.
    *   **DNSPOD Configuration (via GitHub Actions):** Configure using `DOMAIN`, `SUB_DOMAIN`, `SECRETID`, `SECRETKEY`, and optional `PUSHPLUS_TOKEN`.
    *   **DNSCF Configuration (via GitHub Actions):** Configure using `CF_API_TOKEN`, `CF_ZONE_ID`, `CF_DNS_NAME`, and optional `PUSHPLUS_TOKEN`.
*   **PUSHPLUS Notifications:**  Integrates with PUSHPLUS for real-time notification alerts.  ([https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html))

## API Endpoint

The project provides a simple API endpoint to retrieve the top-performing Cloudflare IPs:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

## API Response Example

```
104.16.204.6,104.18.103.125
```

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Powered by

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")