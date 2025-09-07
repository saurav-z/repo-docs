# Accelerate Your Cloudflare CDN with cf-speed-dns

**Optimize your Cloudflare CDN performance by automatically selecting and updating the fastest IP addresses with cf-speed-dns.** Find the original project on GitHub: [ZhiXuanWang/cf-speed-dns](https://github.com/ZhiXuanWang/cf-speed-dns).

## Key Features

*   **Real-time Cloudflare IP Optimization:** Automatically identifies and updates to the best-performing Cloudflare IP addresses every 5 minutes.
*   **Optimized IP Lists:** Access pre-generated lists of optimized Cloudflare IPs.
    *   Top IP List: [https://ip.164746.xyz](https://ip.164746.xyz)
    *   Top IP Interface: [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html)
    *   Top 10 IP Interface: [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **Automated DNS Updates:** Integrates with DNSPOD and DNSCF to automatically update your DNS records with the fastest Cloudflare IPs.
    *   **DNSPOD Integration:** Configure via GitHub Actions with your DOMAIN, SUB\_DOMAIN, SECRETID, SECRETKEY, and PUSHPLUS\_TOKEN.
    *   **DNSCF Integration:** Configure via GitHub Actions with your CF\_API\_TOKEN, CF\_ZONE\_ID, CF\_DNS\_NAME, and PUSHPLUS\_TOKEN.
*   **Push Notifications:** Receive notifications via PUSHPLUS to stay informed about IP updates. [https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html)

## API Endpoint

The following API endpoint provides the top performing IP addresses:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

**Example Response:**

```
104.16.204.6,104.18.103.125
```

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Powered By

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")