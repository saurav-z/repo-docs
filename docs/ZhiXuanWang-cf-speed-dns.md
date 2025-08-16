# cf-speed-dns: Optimize Your Cloudflare CDN with Speed-Tested IPs

Tired of slow website loading times? **cf-speed-dns automatically finds and updates the fastest Cloudflare CDN IPs, delivering optimal performance for your website.**

[View the original repository on GitHub](https://github.com/ZhiXuanWang/cf-speed-dns)

## Key Features

*   **Real-time Cloudflare IP Optimization:** Automatically selects and updates the fastest Cloudflare CDN IPs every 5 minutes.
*   **Optimized IP Lists:** Access pre-sorted IP lists for immediate use:
    *   [Top IPs](https://ip.164746.xyz/ipTop.html) (Default, fastest IPs)
    *   [Top 10 IPs](https://ip.164746.xyz/ipTop10.html)
    *   [Real-time Updated IP List](https://ip.164746.xyz)
*   **DNS Record Updates:**
    *   Automated DNS record updates for DNSPOD using GitHub Actions (requires configuration).
    *   Automated DNS record updates for DNSCF using GitHub Actions (requires configuration).
*   **Push Notifications:** Integrated with PUSHPLUS for real-time notifications on updates.
*   **Easy Integration:** Simple setup with GitHub Actions and straightforward API requests.

## API Endpoint

Retrieve the top performing Cloudflare IPs:

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

## Sponsored by

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")