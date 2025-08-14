# Optimize Your Cloudflare CDN with cf-speed-dns

**Tired of slow website loading times?**  cf-speed-dns automatically finds and updates the fastest Cloudflare IP addresses for optimal performance.  Learn more and contribute at the [original repository](https://github.com/ZhiXuanWang/cf-speed-dns).

## Key Features

*   **Real-time Cloudflare IP Optimization:**  Dynamically selects and updates the best-performing Cloudflare IPs every 5 minutes, ensuring your website leverages the fastest available connections.
*   **Optimized IP Lists:** Access pre-built lists of optimized Cloudflare IPs.
    *   **Top IPs:**  Available at [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) (default)
    *   **Top 10 IPs:** Available at [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **Automated DNS Updates:** Seamlessly integrates with DNSPOD and DNSCF to automatically update your DNS records with the optimized Cloudflare IPs.
    *   **DNSPOD Integration:**  Configure with `DOMAIN`, `SUB_DOMAIN`, `SECRETID`, `SECRETKEY`, and (optionally) `PUSHPLUS_TOKEN` secrets.
    *   **DNSCF Integration:**  Configure with `CF_API_TOKEN`, `CF_ZONE_ID`, `CF_DNS_NAME`, and (optionally) `PUSHPLUS_TOKEN` secrets.
*   **Push Notification Integration:**  Receive notifications via PUSHPLUS to stay informed about updates. (Optional)

## API Endpoint

Get the top optimized IPs with a simple cURL request:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

**Example Response:**

```
104.16.204.6,104.18.103.125
```

## Acknowledgements

This project utilizes and builds upon the work of:

*   [XIU2/CloudflareSpeedTest](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth/cf2dns](https://github.com/ddgth/cf2dns)

## Support

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")