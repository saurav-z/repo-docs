# Find the Fastest Cloudflare IPs with cf-speed-dns

**Optimize your website's performance by automatically identifying and utilizing the quickest Cloudflare IPs for faster content delivery.**  (Original repo: [https://github.com/ZhiXuanWang/cf-speed-dns](https://github.com/ZhiXuanWang/cf-speed-dns))

## Key Features

*   **Cloudflare IP Optimization:**  Finds and provides a list of the fastest Cloudflare IPs, updated every 5 minutes.
*   **Real-time IP Lists:** Access real-time, updated lists of optimized Cloudflare IPs via web interfaces:
    *   [https://ip.164746.xyz](https://ip.164746.xyz)
    *   Top IPs: [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) (default)
    *   Top 10 IPs: [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **DNS Integration (via Forking & Configuration):** Integrates with DNSPOD and DNSCF for automated domain resolution updates.  Requires forking the repository and configuring GitHub Actions with the necessary secrets and variables.
    *   **DNSPOD:** Requires DOMAIN, SUB\_DOMAIN, SECRETID, SECRETKEY, and optionally PUSHPLUS\_TOKEN for notifications.
    *   **DNSCF:** Requires CF\_API\_TOKEN, CF\_ZONE\_ID, CF\_DNS\_NAME, and optionally PUSHPLUS\_TOKEN for notifications.
*   **PUSHPLUS Notification Integration:**  Receive notifications via PUSHPLUS for status updates (configure PUSHPLUS\_TOKEN).
*   **Easy-to-Use API:** Quickly retrieve the top Cloudflare IPs using a simple `curl` command.

## API Endpoint

Use the following endpoint to retrieve the top Cloudflare IPs:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

## API Response Example

The API returns a comma-separated list of the top Cloudflare IPs:

```
104.16.204.6,104.18.103.125
```

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Sponsored by

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")