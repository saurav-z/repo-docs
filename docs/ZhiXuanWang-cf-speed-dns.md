# Optimize Your Cloudflare CDN Performance with cf-speed-dns

**Tired of slow website speeds?**  cf-speed-dns automatically finds and updates the fastest Cloudflare CDN IPs, ensuring optimal performance for your website.

Check out the original repository for more details: [https://github.com/ZhiXuanWang/cf-speed-dns](https://github.com/ZhiXuanWang/cf-speed-dns)

## Key Features:

*   **Real-time Cloudflare IP Optimization:**  Identifies and updates the best-performing Cloudflare IPs every 5 minutes.
*   **Optimized IP Lists:**
    *   Provides a real-time updated list of optimized Cloudflare IPs: [https://ip.164746.xyz](https://ip.164746.xyz)
    *   Offers Top IP endpoints for quick access to the fastest IPs: [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) (default) and [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **Automated DNS Updates:** Integrates with DNS providers to automatically update your DNS records with the fastest Cloudflare IPs.
    *   **DNSPOD Integration:** Configure automated updates through GitHub Actions, requiring `DOMAIN`, `SUB_DOMAIN`, `SECRETID`, `SECRETKEY`, and optionally `PUSHPLUS_TOKEN` secrets.
    *   **DNSCF Integration:**  Utilizes `CF_API_TOKEN`, `CF_ZONE_ID`, and `CF_DNS_NAME` for automatic updates, also supporting `PUSHPLUS_TOKEN` for notifications.
*   **PUSHPLUS Notification Integration:**  Receive instant notifications about updates via PUSHPLUS:  [https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html)

## API Endpoint:

Get the top-performing IPs with a simple API request:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

## API Response Example:

```
104.16.204.6,104.18.103.125
```

## Acknowledgements:

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Powered By

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")