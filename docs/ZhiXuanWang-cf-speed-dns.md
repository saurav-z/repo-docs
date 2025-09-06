## Optimize Your Cloudflare CDN Speed with cf-speed-dns

**Tired of slow Cloudflare speeds?** [cf-speed-dns](https://github.com/ZhiXuanWang/cf-speed-dns) automatically finds and updates the fastest Cloudflare IPs every 5 minutes, ensuring optimal performance for your website.

### Key Features:

*   **Real-time Optimized IP Lists:** Access constantly updated lists of the fastest Cloudflare IPs.
    *   Main list: [https://ip.164746.xyz](https://ip.164746.xyz)
    *   Top IP interface (default): [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html)
    *   Top 10 IP interface: [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **Automated DNS Updates:** Seamlessly integrates with DNSPOD and DNSCF to automatically update your DNS records with the best-performing Cloudflare IPs.
    *   **DNSPOD Integration:** Configure with `DOMAIN`, `SUB_DOMAIN`, `SECRETID`, `SECRETKEY`, and (optional) `PUSHPLUS_TOKEN` via GitHub Actions secrets and variables.
    *   **DNSCF Integration:** Configure with `CF_API_TOKEN`, `CF_ZONE_ID`, `CF_DNS_NAME`, and (optional) `PUSHPLUS_TOKEN` via GitHub Actions secrets and variables.
*   **Push Notifications:** Receive notifications via PUSHPLUS to stay informed about IP updates and potential issues. ([https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html))

### API Access

You can retrieve the top Cloudflare IPs using a simple `curl` command:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

The API returns a comma-separated list of the fastest IPs, for example:

```
104.16.204.6,104.18.103.125
```

### Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

### Supporting Open Source

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")