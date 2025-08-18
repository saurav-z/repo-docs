# cf-speed-dns: Optimize Your Cloudflare CDN with Fast IP Addresses

**Instantly boost your website's speed and performance by automatically selecting the fastest Cloudflare CDN IP addresses every 5 minutes.**  This project, [cf-speed-dns](https://github.com/ZhiXuanWang/cf-speed-dns), provides tools and resources to find and utilize the best-performing Cloudflare IPs for optimal content delivery.

## Key Features:

*   **Cloudflare IP Optimization:** Automatically identifies and updates a list of the fastest Cloudflare CDN IP addresses.
*   **Real-time IP Lists:** Access up-to-date lists of optimized IPs through various interfaces:
    *   [https://ip.164746.xyz](https://ip.164746.xyz)
    *   Top Interface: [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) (default)
    *   Top 10 Interface: [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **Dynamic DNS Integration:**  Integrates with DNS providers like DNSPOD and DNSCF for automatic DNS record updates, ensuring your domain resolves to the fastest IPs.
    *   **DNSPOD Integration:**  Configure using your DOMAIN, SUB_DOMAIN, SECRETID, SECRETKEY, and PUSHPLUS_TOKEN in your Action secrets.
    *   **DNSCF Integration:** Configure using your CF\_API\_TOKEN, CF\_ZONE\_ID, CF\_DNS\_NAME, and PUSHPLUS\_TOKEN in your Action secrets.
*   **Push Notification Support:**  Receive notifications via PUSHPLUS. [https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html)

## API Request Example

Retrieve the top Cloudflare IP addresses using a simple `curl` command:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

## API Response Example

The API returns a comma-separated list of the top-performing Cloudflare IP addresses:

```
104.16.204.6,104.18.103.125
```

## Acknowledgements

This project leverages the work of:

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Support Open Source

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")