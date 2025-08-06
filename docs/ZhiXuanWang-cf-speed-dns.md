# Find the Fastest Cloudflare IP with cf-speed-dns

**Instantly boost your website's speed and performance by utilizing the fastest Cloudflare CDN IPs with `cf-speed-dns`.**

This tool automatically identifies and updates your DNS records with the most optimal Cloudflare IPs, ensuring the best possible performance for your website.  Learn more and contribute on the original repository: [https://github.com/ZhiXuanWang/cf-speed-dns](https://github.com/ZhiXuanWang/cf-speed-dns)

## Key Features:

*   **Cloudflare IP Selection:** Automatically finds and provides the fastest Cloudflare CDN IPs.
*   **Real-time Updates:**  The best IPs are updated every 5 minutes.
*   **Optimized IP Lists:**
    *   Top IPs Interface: [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) (default)
    *   Top 10 IPs Interface: [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **DNS Record Updates:**  Supports automated DNS record updates via:
    *   DNSPOD (requires DOMAIN, SUB\_DOMAIN, SECRETID, SECRETKEY, PUSHPLUS\_TOKEN setup)
    *   DNSCF (requires CF\_API\_TOKEN, CF\_ZONE\_ID, CF\_DNS\_NAME, PUSHPLUS\_TOKEN setup)
*   **PUSHPLUS Notification:** Sends notifications via PUSHPLUS.
*   **Easy-to-Use API:**
    ```javascript
    curl 'https://ip.164746.xyz/ipTop.html'
    ```
    Returns a comma-separated list of optimized IP addresses, e.g., `104.16.204.6,104.18.103.125`

## Credits

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Support
[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")