# Optimize Your Cloudflare CDN with cf-speed-dns

**Tired of slow website loading times?**  cf-speed-dns automatically identifies and updates your DNS settings with the fastest Cloudflare IPs, ensuring optimal performance for your website. Find the original repository [here](https://github.com/ZhiXuanWang/cf-speed-dns).

## Key Features:

*   **Real-time Cloudflare IP Optimization:**  Dynamically updates your DNS records with the fastest Cloudflare IPs, improving website speed and reliability.
*   **Multiple Interface Options:** Choose from a variety of interfaces to access optimized IP lists:
    *   Main List: [https://ip.164746.xyz](https://ip.164746.xyz)
    *   Top IPs (Default): [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html)
    *   Top 10 IPs: [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **DNSPOD & DNSCF Integration:**  Supports automatic DNS record updates for both DNSPOD and DNSCF, streamlining your workflow.
*   **PUSHPLUS Notification Support:** Get notified of updates via PUSHPLUS, keeping you informed about the latest IP changes.  (Link: [https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html))

## API Endpoint:

*   **Request:** `curl 'https://ip.164746.xyz/ipTop.html'`
*   **Response (Example):**
    ```
    104.16.204.6,104.18.103.125
    ```

## Configuration for DNS Providers:

**DNSPOD:**

1.  Fork this project.
2.  In your project's Actions settings, add the following secrets and variables:
    *   `DOMAIN` (e.g., 164746.xyz)
    *   `SUB_DOMAIN` (e.g., dns)
    *   `SECRETID` (Your DNSPOD Secret ID)
    *   `SECRETKEY` (Your DNSPOD Secret Key)
    *   `PUSHPLUS_TOKEN` (Your PUSHPLUS token for notifications)

**DNSCF (Cloudflare):**

1.  Fork this project.
2.  In your project's Actions settings, add the following secrets and variables:
    *   `CF_API_TOKEN` (Your Cloudflare API Token)
    *   `CF_ZONE_ID` (Your Cloudflare Zone ID)
    *   `CF_DNS_NAME` (e.g., dns.164746.xyz)
    *   `PUSHPLUS_TOKEN` (Your PUSHPLUS token for notifications)

## Acknowledgements

This project leverages the work of:

*   [XIU2/CloudflareSpeedTest](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth/cf2dns](https://github.com/ddgth/cf2dns)

## Sponsored by

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")