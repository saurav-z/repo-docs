# Speed Up Your Website with Cloudflare's Fastest IPs: cf-speed-dns

Tired of slow website loading times? **cf-speed-dns automatically finds and updates your Cloudflare DNS records with the fastest IPs available, resulting in improved website performance and a better user experience.**

[View the original project on GitHub](https://github.com/ZhiXuanWang/cf-speed-dns)

## Key Features:

*   **Cloudflare Speed Test Integration:**  Leverages CloudflareSpeedTest to identify and prioritize the fastest Cloudflare IP addresses.
*   **Real-time IP List:** Provides a real-time updated list of optimized IPs. You can view the top optimized IPs here: [https://ip.164746.xyz](https://ip.164746.xyz)
*   **Top IP Endpoint:** Offers a Top IP interface ([https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html)) and Top 10 ([https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html))
*   **DNSPOD Integration:**  Automated domain resolution updates with DNSPOD. Requires setup with DOMAIN, SUB\_DOMAIN, SECRETID, SECRETKEY, and optionally PUSHPLUS\_TOKEN in your Action configuration.
*   **DNSCF Integration:**  Automated domain resolution updates with Cloudflare, using CF\_API\_TOKEN, CF\_ZONE\_ID, CF\_DNS\_NAME, and optionally PUSHPLUS\_TOKEN in your Action configuration.
*   **PUSHPLUS Notification:**  Integrates with PUSHPLUS for notification alerts.  [https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html)

## API Endpoint

The API provides the Top IPs in a comma-separated format:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

## API Response Example:

```javascript
104.16.204.6,104.18.103.125
```

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Advertisement

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")