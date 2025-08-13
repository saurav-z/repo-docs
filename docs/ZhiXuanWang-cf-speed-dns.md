# Optimize Your Cloudflare CDN Performance with cf-speed-dns

**Tired of slow website loading speeds?** `cf-speed-dns` helps you find the fastest Cloudflare IP addresses for optimal performance and automatic DNS updates.  ([See the original repository here](https://github.com/ZhiXuanWang/cf-speed-dns))

## Key Features:

*   **Real-time Cloudflare IP Optimization:**  Identifies and provides a list of the fastest Cloudflare IP addresses, updated every 5 minutes.
*   **Optimized IP Lists:** Access pre-sorted lists of optimal IPs for improved speeds:
    *   **Top Interface:**  `https://ip.164746.xyz/ipTop.html` (default) - Provides the top-performing IPs.
    *   **Top 10 Interface:** `https://ip.164746.xyz/ipTop10.html` - Provides the top 10 performing IPs.
*   **Automated DNS Updates:** Integrates with DNSPOD and DNSCF for automatic DNS record updates with the best performing IPs.
    *   **DNSPOD Integration:** Configure via GitHub Actions with your domain, subdomain, SECRETID, SECRETKEY, and (optional) PUSHPLUS_TOKEN for notifications.
    *   **DNSCF Integration:** Configure via GitHub Actions using your CF\_API\_TOKEN, CF\_ZONE\_ID, CF\_DNS\_NAME, and (optional) PUSHPLUS\_TOKEN for notifications.
*   **PUSHPLUS Notification Support:**  Receive notifications via PUSHPLUS for updates and events.

## How to Use the API

You can retrieve the top-performing IPs using a simple `curl` command:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

The API returns a comma-separated list of the best Cloudflare IP addresses:

```
104.16.204.6,104.18.103.125
```

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Powered by DartNode
[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")