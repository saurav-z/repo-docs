# Optimize Your Cloudflare CDN with cf-speed-dns

**Quickly find and automatically update your Cloudflare DNS records with the fastest IP addresses using `cf-speed-dns`!**  You can find the original repository [here](https://github.com/ZhiXuanWang/cf-speed-dns).

## Key Features:

*   **Cloudflare Speed Test Integration:**  Automatically selects and updates your Cloudflare DNS with the fastest IPs based on real-time speed tests.
*   **Real-time IP List:** Access a live list of optimized Cloudflare IP addresses.
*   **Top IP Interfaces:**  Use pre-built interfaces to retrieve the top performing IPs (default and Top 10 options available).
*   **DNSPOD Integration:** Easily integrate with DNSPOD for automated DNS record updates. Configure via GitHub Actions.
    *   Supports custom domain and subdomain configuration.
*   **DNSCF Integration:**  Integrate with Cloudflare directly for automated DNS record updates.  Configure via GitHub Actions.
    *   Requires Cloudflare API Token and Zone ID.
    *   Supports custom DNS name configuration.
*   **PUSHPLUS Notifications:**  Receive notifications about DNS updates via PUSHPLUS.

## API Endpoints:

*   **Top IPs (Default):** `https://ip.164746.xyz/ipTop.html`  (returns comma-separated IPs)
*   **Top 10 IPs:** `https://ip.164746.xyz/ipTop10.html`

## Example API Request:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

## Example API Response:

```
104.16.204.6,104.18.103.125
```

## Acknowledgements:

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Powered by:

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")