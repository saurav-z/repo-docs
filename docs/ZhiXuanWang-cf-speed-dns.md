# Find the Fastest Cloudflare IPs with cf-speed-dns

**Tired of slow Cloudflare performance?**  This tool automatically identifies and updates the fastest Cloudflare IPs for optimal speed and performance.  

[Visit the original repository](https://github.com/ZhiXuanWang/cf-speed-dns) for more details and to contribute.

## Key Features of cf-speed-dns:

*   **Real-time Cloudflare IP Optimization:**  Automatically identifies and prioritizes the fastest Cloudflare IPs.
*   **Updated IP Lists:**  Provides access to up-to-date lists of optimized IPs via web interfaces:
    *   [Main List](https://ip.164746.xyz)
    *   [Top IPs](https://ip.164746.xyz/ipTop.html)
    *   [Top 10 IPs](https://ip.164746.xyz/ipTop10.html)
*   **Automated DNS Updates:**  Integrates with popular DNS providers for automatic updates:
    *   **DNSPOD Integration:** Supports automatic DNS record updates via GitHub Actions (requires configuration of `DOMAIN`, `SUB_DOMAIN`, `SECRETID`, `SECRETKEY`, and optionally `PUSHPLUS_TOKEN`).
    *   **DNSCF Integration:** Supports automatic DNS record updates via GitHub Actions (requires configuration of `CF_API_TOKEN`, `CF_ZONE_ID`, `CF_DNS_NAME`, and optionally `PUSHPLUS_TOKEN`).
*   **PUSHPLUS Notification Support:**  Optionally receive notifications about IP updates via [PUSHPLUS](https://www.pushplus.plus/push1.html).

## API Endpoint

Access the top Cloudflare IPs via a simple API endpoint:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

### API Response Example:

```text
104.16.204.6,104.18.103.125
```

## Acknowledgements

Thanks to the following projects for inspiration and contributions:

*   [XIU2/CloudflareSpeedTest](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth/cf2dns](https://github.com/ddgth/cf2dns)

## Powered By
[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")