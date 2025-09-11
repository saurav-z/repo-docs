# Find the Fastest Cloudflare CDN IPs with cf-speed-dns

**Instantly optimize your Cloudflare performance by finding and using the fastest CDN IPs with `cf-speed-dns`!**  [Check out the original repository](https://github.com/ZhiXuanWang/cf-speed-dns).

## Key Features

*   **Automated IP Optimization:**  Finds and provides a list of the best-performing Cloudflare CDN IPs every 5 minutes.
*   **Real-time IP Lists:** Access up-to-date lists of optimized IPs via these interfaces:
    *   [https://ip.164746.xyz](https://ip.164746.xyz) (Main List)
    *   [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) (Top IPs - Default)
    *   [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html) (Top 10 IPs)
*   **DNSPOD Integration:**  Integrates with DNSPOD to automatically update your DNS records with the fastest IPs (requires configuration).
*   **DNSCF Integration:** Integrate with DNSCF to automatically update your DNS records with the fastest IPs (requires configuration).
*   **PUSHPLUS Notifications:**  Receive notifications through PUSHPLUS ([https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html)).

## API Request Example

Here's how to fetch the top-performing IPs:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

## API Response Example

The API returns a comma-separated list of the fastest IP addresses:

```
104.16.204.6,104.18.103.125
```

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Advertisement

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")