# Optimize Your Cloudflare CDN with cf-speed-dns: Find the Fastest IPs!

Tired of slow website loading times? **cf-speed-dns automatically identifies and updates the fastest Cloudflare CDN IPs to optimize your website's performance.**  Get the latest updates and learn more about this powerful tool on the original repository: [https://github.com/ZhiXuanWang/cf-speed-dns](https://github.com/ZhiXuanWang/cf-speed-dns).

## Key Features of cf-speed-dns

*   **Real-time Cloudflare IP Optimization:** Leverages CloudflareSpeedTest to identify and select the most performant Cloudflare CDN IPs every 5 minutes.
*   **Up-to-Date IP Lists:** Access updated lists of optimized IPs via the provided interfaces:
    *   [https://ip.164746.xyz](https://ip.164746.xyz)
    *   Top Interface (default): [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html)
    *   Top 10 Interface: [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **DNSPOD & DNSCF Integration:**  Supports real-time domain name resolution updates with DNSPOD and DNSCF services via GitHub Actions.
    *   **DNSPOD Configuration:** Configure your GitHub Actions secrets and variables with DOMAIN, SUB_DOMAIN, SECRETID, SECRETKEY, and PUSHPLUS_TOKEN.
    *   **DNSCF Configuration:** Configure your GitHub Actions secrets and variables with CF_API_TOKEN, CF_ZONE_ID, CF_DNS_NAME, and PUSHPLUS_TOKEN.
*   **PUSHPLUS Notification Support:** Get instant notifications about IP updates via PUSHPLUS.  Learn more: [https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html)

## API Usage

Retrieve the top optimized IPs using a simple `curl` command:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

The API returns a comma-separated list of the fastest Cloudflare IPs, for example:

```
104.16.204.6,104.18.103.125
```

## Acknowledgements

This project is built upon the work of:

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Advertisement

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")