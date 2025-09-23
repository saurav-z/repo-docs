# Find the Fastest Cloudflare IP with cf-speed-dns

**Tired of slow Cloudflare speeds?**  This project automatically identifies and updates your Cloudflare DNS with the fastest IP addresses for optimal performance. This project is a fork of [CloudflareSpeedTest](https://github.com/XIU2/CloudflareSpeedTest), which uses CloudflareSpeedTest to find the best Cloudflare CDN IPs for you, then uses other tools to update your DNS records.

## Key Features

*   **Real-Time IP Optimization:**  Automatically selects and updates Cloudflare IP addresses every 5 minutes based on speed and latency.
*   **Optimized IP Lists:** Provides access to optimized IP lists via:
    *   Top interface: [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) (default)
    *   Top 10 interface: [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **DNS Integration:** Supports automatic DNS record updates for your domain, including:
    *   DNSPOD integration (fork required, configure with DOMAIN, SUB\_DOMAIN, SECRETID, SECRETKEY, and PUSHPLUS\_TOKEN in Actions secrets).
    *   DNSCF integration (fork required, configure with CF\_API\_TOKEN, CF\_ZONE\_ID, CF\_DNS\_NAME, and PUSHPLUS\_TOKEN in Actions secrets).
*   **Notification Support:**  Sends notifications using PUSHPLUS for status updates ([https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html)).

## How it Works

The project leverages CloudflareSpeedTest to identify the fastest Cloudflare IP addresses and then updates your DNS records.

## API Endpoint (Example)

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

## API Response (Example)

```
104.16.204.6,104.18.103.125
```

## Getting Started

1.  **Fork the repository** to customize DNS updates.
2.  **Configure** your Actions secrets and variables according to your DNS provider (DNSPOD or DNSCF).
3.  **Enable** Github Actions.

For more detailed instructions and the original source, please visit the original repository:  [https://github.com/ZhiXuanWang/cf-speed-dns](https://github.com/ZhiXuanWang/cf-speed-dns)

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Support

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")