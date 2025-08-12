# Boost Your Website Speed with Optimized Cloudflare IPs: cf-speed-dns

**Tired of slow website loading times?**  cf-speed-dns provides a streamlined solution to identify and utilize the fastest Cloudflare CDN IPs for optimal performance.

**[View the original project on GitHub](https://github.com/ZhiXuanWang/cf-speed-dns)**

## Key Features:

*   **Real-time Cloudflare IP Optimization:**  Continuously identifies and provides the best-performing Cloudflare CDN IPs, updating every 5 minutes.
*   **Optimized IP Lists:**  Access optimized IP lists through readily available interfaces:
    *   **Top IPs:**  [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) (Default)
    *   **Top 10 IPs:** [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
    *   **Full List:**  [https://ip.164746.xyz](https://ip.164746.xyz)
*   **Automated DNS Updates:** Integrates with DNS providers for automatic updates:
    *   **DNSPOD Integration:**  Configure using your DOMAIN, SUB\_DOMAIN, SECRETID, SECRETKEY, and PUSHPLUS\_TOKEN.
    *   **DNSCF Integration:**  Configure using your CF\_API\_TOKEN, CF\_ZONE\_ID, CF\_DNS\_NAME, and PUSHPLUS\_TOKEN.
*   **Push Notification Support:**  Receive instant updates and notifications via PUSHPLUS.  [https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html)

## API Endpoint:

*   **Get Top IPs:**

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

*   **Response Example:**

```
104.16.204.6,104.18.103.125
```

## Acknowledgements:

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Sponsored by:

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")