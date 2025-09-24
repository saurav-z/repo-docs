# Find the Fastest Cloudflare IPs with cf-speed-dns

**Tired of slow website loading times?** cf-speed-dns automatically finds and updates the best Cloudflare IP addresses for optimal performance.

[View the original repository on GitHub](https://github.com/ZhiXuanWang/cf-speed-dns)

## Key Features of cf-speed-dns

*   **Real-time Cloudflare IP Optimization:** Automatically selects the fastest Cloudflare IPs every 5 minutes.
*   **Optimized IP Lists:** Provides pre-built lists of optimized IPs via web interfaces.
    *   Main List: [https://ip.164746.xyz](https://ip.164746.xyz)
    *   Top IPs (Default): [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html)
    *   Top 10 IPs: [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **Automated DNS Updates:** Integrates with DNS providers to automatically update your DNS records with the best performing Cloudflare IPs.  Supports DNSPOD and DNSCF.
    *   **DNSPOD Integration:**  Requires configuration of secrets and variables in GitHub Actions, including `DOMAIN`, `SUB_DOMAIN`, `SECRETID`, `SECRETKEY`, and optionally `PUSHPLUS_TOKEN` for notifications.
    *   **DNSCF Integration:** Requires configuration of secrets and variables in GitHub Actions, including `CF_API_TOKEN`, `CF_ZONE_ID`, `CF_DNS_NAME`, and optionally `PUSHPLUS_TOKEN` for notifications.
*   **PUSHPLUS Notifications:** Optional integration with PUSHPLUS for real-time notifications about IP updates.

## API Endpoint Example

You can access the top IPs via this API endpoint:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

**Example Response:**

```
104.16.204.6,104.18.103.125
```

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Advertisement
[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")