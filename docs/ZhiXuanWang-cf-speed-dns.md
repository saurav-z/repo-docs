# Optimize Your Cloudflare CDN Speed with cf-speed-dns

**Tired of slow website loading times?**  This project helps you identify and utilize the fastest Cloudflare IPs for optimal website performance. ([Original Repository](https://github.com/ZhiXuanWang/cf-speed-dns))

## Key Features:

*   **Real-time Cloudflare IP Optimization:** Automatically identifies and lists the fastest Cloudflare CDN IPs using CloudflareSpeedTest.
*   **Easy-to-Use IP Lists:** Provides readily available lists of optimized IPs:
    *   Main List: [https://ip.164746.xyz](https://ip.164746.xyz)
    *   Top IP List (default): [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html)
    *   Top 10 IP List: [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **Automated DNS Updates:** Integrates with DNS providers for seamless IP updates:
    *   **DNSPOD:**  Supports automated DNS record updates. Configure through GitHub Actions with your DOMAIN, SUB\_DOMAIN, SECRETID, SECRETKEY, and (optional) PUSHPLUS\_TOKEN for notifications.
    *   **DNSCF:**  Supports automated Cloudflare DNS record updates.  Configure through GitHub Actions with your CF\_API\_TOKEN, CF\_ZONE\_ID, CF\_DNS\_NAME, and (optional) PUSHPLUS\_TOKEN for notifications.
*   **Push Notifications:**  Integrates with PUSHPLUS for timely notifications on IP updates. ([https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html))

## API Usage

Get the top performing IPs with a simple `curl` command:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

Example response:

```
104.16.204.6,104.18.103.125
```

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Advertisement

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")