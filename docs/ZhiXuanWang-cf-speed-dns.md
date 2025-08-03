# Optimize Your Cloudflare CDN Speed with cf-speed-dns

**Tired of slow website loading times?** `cf-speed-dns` automatically finds and updates the fastest Cloudflare CDN IP addresses, ensuring optimal performance for your website.  ([See the original repository](https://github.com/ZhiXuanWang/cf-speed-dns))

## Key Features:

*   **Real-time Cloudflare IP Optimization:** Automatically selects and updates the best-performing Cloudflare IPs every 5 minutes.
*   **Optimized IP Lists:** Provides access to top-performing IP lists, including:
    *   [https://ip.164746.xyz](https://ip.164746.xyz) (Real-time updated list)
    *   [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) (Top IPs - default)
    *   [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html) (Top 10 IPs)
*   **DNSPOD Integration:**  Supports real-time domain name resolution updates via DNSPOD. (Requires configuration of `DOMAIN`, `SUB_DOMAIN`, `SECRETID`, `SECRETKEY`, and optionally `PUSHPLUS_TOKEN` in Actions secrets and variables).
*   **DNSCF Integration:** Provides support for real-time domain name resolution updates via DNSCF. (Requires configuration of `CF_API_TOKEN`, `CF_ZONE_ID`, `CF_DNS_NAME`, and optionally `PUSHPLUS_TOKEN` in Actions secrets and variables).
*   **Push Notifications:** Integrates with PUSHPLUS for notification alerts. ([https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html))

## API Endpoint

You can retrieve the optimized IP list via this endpoint:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

## API Response Example

The API returns a comma-separated list of the top-performing Cloudflare IPs:

```
104.16.204.6,104.18.103.125
```

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Support

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")