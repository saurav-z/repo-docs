# Optimize Your Cloudflare CDN with cf-speed-dns

**Instantly boost your website's performance by finding and utilizing the fastest Cloudflare CDN IP addresses with `cf-speed-dns`.**  ([View the original repository](https://github.com/ZhiXuanWang/cf-speed-dns))

This project helps you identify and automatically update your Cloudflare DNS records with the optimal IP addresses for the best speed and performance.

## Key Features of cf-speed-dns:

*   **Real-time Cloudflare IP Optimization:**  Identifies and provides a list of the fastest Cloudflare IPs.
*   **Optimized IP Lists:** Access pre-built lists of top-performing IPs, including:
    *   [Top IPs List](https://ip.164746.xyz/ipTop.html)
    *   [Top 10 IPs List](https://ip.164746.xyz/ipTop10.html)
    *   [Updated IP List](https://ip.164746.xyz)
*   **Automated DNS Updates:** Seamlessly integrates with DNSPOD and DNSCF to automatically update your DNS records with the best-performing IPs.
    *   **DNSPOD Integration:** Configure using `DOMAIN`, `SUB_DOMAIN`, `SECRETID`, `SECRETKEY`, and `PUSHPLUS_TOKEN` secrets in your GitHub Actions workflow.
    *   **DNSCF Integration:** Configure using `CF_API_TOKEN`, `CF_ZONE_ID`, `CF_DNS_NAME`, and `PUSHPLUS_TOKEN` secrets in your GitHub Actions workflow.
*   **Push Notifications:**  Receive updates and notifications via PUSHPLUS.  [Learn more about PUSHPLUS](https://www.pushplus.plus/push1.html).

## API Endpoint Example

Retrieve the top-performing IPs using a simple API request:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

The API returns a comma-separated list of optimized IP addresses.

```
104.16.204.6,104.18.103.125
```

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Powered by

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")