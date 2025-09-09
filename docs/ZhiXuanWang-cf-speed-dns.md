## Optimize Your Cloudflare CDN Speed with cf-speed-dns

Tired of slow website loading times? **cf-speed-dns helps you automatically find and use the fastest Cloudflare IPs for optimal performance.** (See the original repo [here](https://github.com/ZhiXuanWang/cf-speed-dns).)

### Key Features:

*   **Real-time Cloudflare IP Optimization:**  Identifies and provides the best-performing Cloudflare IPs based on latency and speed.
*   **Optimized IP Lists:** Access lists of top-performing IPs for immediate use, including:
    *   [https://ip.164746.xyz](https://ip.164746.xyz) (Updated IP list)
    *   [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) (Top IPs - Default)
    *   [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html) (Top 10 IPs)
*   **Automated DNS Updates:** Integrates with DNSPOD and DNSCF to automatically update your DNS records with the optimal Cloudflare IPs. (Requires configuration, see below).
*   **PUSHPLUS Notification:**  Receive notifications via PUSHPLUS to stay informed about IP updates. (Requires configuration, see below).

### Configuration for Automated DNS Updates:

You can configure cf-speed-dns to automatically update your DNS records with the optimal Cloudflare IPs using either DNSPOD or DNSCF.

**DNSPOD Configuration (Fork the project):**

1.  In your forked repository's Actions secrets and variables, add the following:
    *   `DOMAIN`: Your domain name (e.g., `164746.xyz`)
    *   `SUB_DOMAIN`:  Your subdomain (e.g., `dns`)
    *   `SECRETID`: Your DNSPOD Secret ID.
    *   `SECRETKEY`: Your DNSPOD Secret Key.
    *   `PUSHPLUS_TOKEN`: Your PUSHPLUS token for notifications.

**DNSCF Configuration (Fork the project):**

1.  In your forked repository's Actions secrets and variables, add the following:
    *   `CF_API_TOKEN`: Your Cloudflare API Token.
    *   `CF_ZONE_ID`:  Your Cloudflare Zone ID.
    *   `CF_DNS_NAME`: Your DNS record name (e.g., `dns.164746.xyz`).
    *   `PUSHPLUS_TOKEN`: Your PUSHPLUS token for notifications.

### API Request Example:

```javascript
curl 'https://ip.164746.xyz/ipTop.html'
```

### API Response Example:

```javascript
104.16.204.6,104.18.103.125
```

### Acknowledgements:

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

### Powered by DartNode

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")