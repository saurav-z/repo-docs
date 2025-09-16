# Optimize Your Cloudflare CDN with cf-speed-dns

**Tired of slow Cloudflare performance?** cf-speed-dns is a powerful tool that helps you find and automatically use the fastest Cloudflare IP addresses for optimal speed and reliability.

**Check out the original repo for the latest updates: [https://github.com/ZhiXuanWang/cf-speed-dns](https://github.com/ZhiXuanWang/cf-speed-dns)**

## Key Features of cf-speed-dns:

*   **Real-time Cloudflare IP Optimization:** Identifies and provides a list of the fastest Cloudflare IP addresses every 5 minutes.
*   **Optimized IP List Access:** Access top-performing IP lists through easy-to-use interfaces:
    *   [https://ip.164746.xyz](https://ip.164746.xyz) (Main List)
    *   [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) (Top IP List - Default)
    *   [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html) (Top 10 IP List)
*   **Automated DNS Updates:** Integrates with DNSPOD and DNSCF to automatically update your DNS records with the optimized IP addresses.
*   **Customizable DNS Integration:** Easily configure DNS updates using GitHub Actions.
*   **Push Notification Support:** Receive notifications via PUSHPLUS to stay informed about IP updates.
*   **Simple API Access:** Quickly retrieve the top-performing IP addresses using a simple API endpoint.

## API Endpoint

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

## API Response Example

```
104.16.204.6,104.18.103.125
```

## Configuration for DNS Updates

To enable automatic DNS updates, you can configure the following secrets and variables within your GitHub Actions workflow:

*   **DNSPOD:**
    *   `DOMAIN`: Your domain (e.g., `164746.xyz`)
    *   `SUB_DOMAIN`: Your subdomain (e.g., `dns`)
    *   `SECRETID`
    *   `SECRETKEY`
    *   `PUSHPLUS_TOKEN` (Optional for push notifications)

*   **DNSCF:**
    *   `CF_API_TOKEN`
    *   `CF_ZONE_ID`
    *   `CF_DNS_NAME`: Your DNS record name (e.g., `dns.164746.xyz`)
    *   `PUSHPLUS_TOKEN` (Optional for push notifications)

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Advertisement

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")