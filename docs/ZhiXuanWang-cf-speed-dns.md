## Optimize Your Cloudflare CDN Speed with cf-speed-dns

**Tired of slow website loading times?**  cf-speed-dns is your solution for automatically finding and utilizing the fastest Cloudflare IP addresses for optimal performance.  Get the most out of your Cloudflare CDN with this powerful tool!

[Check out the original repository](https://github.com/ZhiXuanWang/cf-speed-dns) for more details and to get started.

### Key Features

*   **Real-time Cloudflare IP Optimization:**  Continuously identifies and updates the best-performing Cloudflare IP addresses.
*   **Optimized IP Lists:** Access pre-sorted lists of the fastest IPs via the following interfaces:
    *   [https://ip.164746.xyz](https://ip.164746.xyz)
    *   Top Interface: [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) (Default)
    *   Top 10 Interface: [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **Automated DNS Record Updates:**  Integrates with DNSPOD and DNSCF to automatically update your DNS records with the optimal Cloudflare IPs.
    *   Supports automated updates using GitHub Actions.
    *   Easily configure your settings using secrets and variables within your Action configuration.
*   **Push Notifications:**  Receive notifications on IP updates and status via PUSHPLUS.
*   **Simple API Interface:** Easily access the fastest Cloudflare IPs with a simple `curl` command and a straightforward response.

### API Endpoint Example

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

### API Response Example

```
104.16.204.6,104.18.103.125
```

### Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

### Advertisement

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")