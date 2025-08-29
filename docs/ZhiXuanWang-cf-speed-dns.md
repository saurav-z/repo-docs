# Find the Fastest Cloudflare IPs with cf-speed-dns

Tired of slow website loading times? **cf-speed-dns automatically identifies and provides the quickest Cloudflare IPs to optimize your internet experience.** This project leverages CloudflareSpeedTest to find the best-performing IPs and allows you to integrate them with popular DNS services.  For more details, see the original repository:  [ZhiXuanWang/cf-speed-dns](https://github.com/ZhiXuanWang/cf-speed-dns).

## Key Features

*   **Real-time Cloudflare IP Optimization:**  Uses CloudflareSpeedTest to identify and provide the fastest Cloudflare IPs every 5 minutes.
*   **Optimized IP Lists:** Access pre-generated lists of optimized IPs via HTTP endpoints.
    *   **Top IPs:** [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) (Default)
    *   **Top 10 IPs:** [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **DNSPOD Integration:**  Automated DNS record updates via DNSPOD using GitHub Actions.  Requires configuration of:
    *   `DOMAIN`
    *   `SUB_DOMAIN`
    *   `SECRETID`
    *   `SECRETKEY`
    *   `PUSHPLUS_TOKEN` (for notifications)
*   **DNSCF Integration:** Automated DNS record updates via DNSCF using GitHub Actions. Requires configuration of:
    *   `CF_API_TOKEN`
    *   `CF_ZONE_ID`
    *   `CF_DNS_NAME`
    *   `PUSHPLUS_TOKEN` (for notifications)
*   **PUSHPLUS Notification Support:** Receive instant notifications about IP updates via [PUSHPLUS](https://www.pushplus.plus/push1.html).

## API Endpoint

Retrieve the top optimized Cloudflare IPs with a simple `curl` command:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

## API Response Example

The API returns a comma-separated list of the fastest IP addresses:

```
104.16.204.6,104.18.103.125
```

## Acknowledgements

*   [XIU2/CloudflareSpeedTest](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth/cf2dns](https://github.com/ddgth/cf2dns)

##  Support

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")