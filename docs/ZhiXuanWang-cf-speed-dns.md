# Optimize Your Cloudflare CDN with cf-speed-dns

**Instantly boost your website's performance by utilizing the fastest Cloudflare CDN IPs with cf-speed-dns!** [See the original project on GitHub](https://github.com/ZhiXuanWang/cf-speed-dns).

## Key Features:

*   **Optimized Cloudflare IP Selection:** Automatically identifies and provides the fastest Cloudflare CDN IPs.
*   **Real-Time IP List Updates:**  Access a constantly updated list of optimized IPs. (Hosted at: [https://ip.164746.xyz](https://ip.164746.xyz))
*   **Top IP Interfaces:** Quickly access the top-performing IPs.
    *   Top Interface (Default): [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html)
    *   Top 10 Interface: [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **DNSPOD & DNSCF Integration:** Easily integrate with DNSPOD and DNSCF for real-time domain name resolution updates. Requires setting up Actions secrets and variables (see below).
*   **PUSHPLUS Notification Support:**  Receive notifications via PUSHPLUS. ([https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html))

## Configuration for Domain Name Resolution

To enable domain name resolution updates, you'll need to fork this project and configure the following environment variables within GitHub Actions:

*   **DNSPOD Configuration (using Actions secrets and variables):**
    *   `DOMAIN`: Your domain name (e.g., `164746.xyz`)
    *   `SUB_DOMAIN`: Your subdomain (e.g., `dns`)
    *   `SECRETID`: Your DNSPOD Secret ID
    *   `SECRETKEY`: Your DNSPOD Secret Key
    *   `PUSHPLUS_TOKEN` (Optional): Your PUSHPLUS token for notifications.

*   **DNSCF Configuration (using Actions secrets and variables):**
    *   `CF_API_TOKEN`: Your Cloudflare API Token
    *   `CF_ZONE_ID`: Your Cloudflare Zone ID
    *   `CF_DNS_NAME`: Your DNS record name (e.g., `dns.164746.xyz`)
    *   `PUSHPLUS_TOKEN` (Optional): Your PUSHPLUS token for notifications.

## API Endpoint

Get the top performing IPs:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

## API Response Example

```
104.16.204.6,104.18.103.125
```

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Support & Hosting

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")