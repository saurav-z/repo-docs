# Optimize Your Cloudflare CDN with cf-speed-dns

**Instantly find and utilize the fastest Cloudflare CDN IPs with cf-speed-dns, optimizing your website's performance and speed.** [Explore the original repository](https://github.com/ZhiXuanWang/cf-speed-dns) for more details.

## Key Features of cf-speed-dns:

*   **Real-time Cloudflare IP Optimization:** Leverages CloudflareSpeedTest to identify and provide the best-performing Cloudflare IPs, updated every 5 minutes.
*   **Easy-to-Use IP Lists:** Access optimized IP lists through dedicated interfaces:
    *   [https://ip.164746.xyz](https://ip.164746.xyz) (Real-time updated IP list)
    *   [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) (Top IPs - default)
    *   [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html) (Top 10 IPs)
*   **Automated DNS Updates:** Integrates with DNSPOD and DNSCF for real-time domain name resolution updates.  Supports automated updates via GitHub Actions.
    *   **DNSPOD Integration:** Configure with `DOMAIN`, `SUB_DOMAIN`, `SECRETID`, `SECRETKEY`, and optionally `PUSHPLUS_TOKEN`.
    *   **DNSCF Integration:** Configure with `CF_API_TOKEN`, `CF_ZONE_ID`, `CF_DNS_NAME`, and optionally `PUSHPLUS_TOKEN`.
*   **Push Notifications:** Supports PUSHPLUS for instant notification of IP updates.

## API Request and Response

**Request:**

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

**Response (Example):**

```
104.16.204.6,104.18.103.125
```

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Powered By

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")