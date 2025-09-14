# Optimize Your Cloudflare CDN with cf-speed-dns: Find the Fastest IPs

Tired of slow website loading times? **cf-speed-dns helps you identify and utilize the fastest Cloudflare CDN IPs for optimal performance.** This project automatically finds the best-performing Cloudflare IPs and allows you to easily update your DNS records for improved speed and reliability.

[**See the original repository on GitHub**](https://github.com/ZhiXuanWang/cf-speed-dns)

## Key Features of cf-speed-dns:

*   **Real-time Cloudflare IP Optimization:** Automatically identifies and updates the fastest Cloudflare IPs every 5 minutes using CloudflareSpeedTest.
*   **Optimized IP Lists:** Provides access to optimized IP lists, including a top-performing IP list (default) and a top 10 IP list for increased flexibility.
    *   **Top Interface (Default):** [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html)
    *   **Top 10 Interface:** [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **Automated DNS Updates:** Integrates with DNSPOD and DNSCF for automatic DNS record updates. Configure using GitHub Actions secrets and variables.
    *   **DNSPOD Integration:** Configure with `DOMAIN`, `SUB_DOMAIN`, `SECRETID`, `SECRETKEY`, and `PUSHPLUS_TOKEN`.
    *   **DNSCF Integration:** Configure with `CF_API_TOKEN`, `CF_ZONE_ID`, `CF_DNS_NAME`, and `PUSHPLUS_TOKEN`.
*   **PUSHPLUS Notification:** Sends notifications via PUSHPLUS to keep you informed of updates.
*   **Easy to Use API:** Simple API endpoint to retrieve the optimized Cloudflare IPs.

## API Endpoint

You can fetch the top-performing Cloudflare IPs using this API endpoint:

```javascript
curl 'https://ip.164746.xyz/ipTop.html'
```

## API Response Example

The API response returns a comma-separated list of the fastest Cloudflare IPs:

```javascript
104.16.204.6,104.18.103.125
```

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Advertisement

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")