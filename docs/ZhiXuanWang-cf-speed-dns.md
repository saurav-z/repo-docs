## Optimize Your Cloudflare CDN Performance with cf-speed-dns

**Tired of slow website speeds?** cf-speed-dns automatically finds and updates the fastest Cloudflare IP addresses for optimal performance. [See the original repository](https://github.com/ZhiXuanWang/cf-speed-dns).

### Key Features

*   **Real-time Cloudflare IP Optimization:** Automatically identifies and provides the quickest Cloudflare IP addresses for your needs, updated every 5 minutes.
*   **Easy-to-Use IP Lists:** Access lists of optimized IPs through readily available interfaces:
    *   **Main List:** [https://ip.164746.xyz](https://ip.164746.xyz)
    *   **Top IPs (Default):** [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html)
    *   **Top 10 IPs:** [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **Automated DNS Updates:** Integrates with DNSPOD and DNSCF for automatic domain name resolution updates using your chosen IP.
*   **Notification Support:** Get notified of IP changes and updates via PUSHPLUS.
*   **DNSPOD Integration:** Configure via Actions secrets and variables: `DOMAIN`, `SUB_DOMAIN`, `SECRETID`, `SECRETKEY`, `PUSHPLUS_TOKEN`.
*   **DNSCF Integration:** Configure via Actions secrets and variables: `CF_API_TOKEN`, `CF_ZONE_ID`, `CF_DNS_NAME`, `PUSHPLUS_TOKEN`.

### API Endpoint

Retrieve the top performing IPs with a simple `curl` command:

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

### Powered by

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")