# Optimize Your Cloudflare CDN with cf-speed-dns

Tired of slow website loading speeds? **cf-speed-dns automatically finds and configures the fastest Cloudflare CDN IP addresses for optimal performance.**

[View the original project on GitHub](https://github.com/ZhiXuanWang/cf-speed-dns)

## Key Features of cf-speed-dns:

*   **Real-time Cloudflare IP Optimization:** Automatically selects and updates the fastest Cloudflare IP addresses every 5 minutes, ensuring the best possible speed and performance.
*   **Multiple IP List Endpoints:**
    *   Provides optimized IP lists via endpoints for easy integration:
        *   Main List: [https://ip.164746.xyz](https://ip.164746.xyz)
        *   Top IPs: [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) (Default)
        *   Top 10 IPs: [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **DNSPOD & DNSCF Integration:** Seamlessly integrates with DNSPOD and DNSCF services for automated domain name resolution using the optimized Cloudflare IPs.  Configuration involves setting up Actions secrets and variables within your forked repository.
    *   **DNSPOD Configuration:** Requires `DOMAIN`, `SUB_DOMAIN`, `SECRETID`, `SECRETKEY`, and optionally `PUSHPLUS_TOKEN`.
    *   **DNSCF Configuration:** Requires `CF_API_TOKEN`, `CF_ZONE_ID`, `CF_DNS_NAME`, and optionally `PUSHPLUS_TOKEN`.
*   **PUSHPLUS Notification Support:** Sends notifications via PUSHPLUS when the optimized IPs are updated.
*   **Easy API Access:** Simple REST API to retrieve the top performing IPs:
    ```bash
    curl 'https://ip.164746.xyz/ipTop.html'
    ```
    Returns a comma-separated list of the fastest IPs, for example:
    ```
    104.16.204.6,104.18.103.125
    ```

## Acknowledgements

*   [XIU2/CloudflareSpeedTest](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth/cf2dns](https://github.com/ddgth/cf2dns)

##  Powered by

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")