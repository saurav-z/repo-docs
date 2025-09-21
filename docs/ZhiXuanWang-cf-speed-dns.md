# cf-speed-dns: Automatically Find and Utilize the Fastest Cloudflare IPs

**Tired of slow website loading speeds?**  cf-speed-dns automatically identifies and updates your Cloudflare DNS with the quickest IPs, optimizing your website's performance. ([See the original repo](https://github.com/ZhiXuanWang/cf-speed-dns))

## Key Features:

*   **Real-time Cloudflare IP Optimization:**  Continuously updates your DNS with the fastest Cloudflare IPs, improving website loading times.
*   **Pre-built IP Lists:** Access pre-compiled lists of optimized IPs.
    *   Top IPs: [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html)
    *   Top 10 IPs: [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
    *   Full List: [https://ip.164746.xyz](https://ip.164746.xyz)
*   **DNSPOD and DNSCF Integration:**  Automated DNS updates for DNSPOD and DNSCF.  Easy configuration via GitHub Actions.
*   **PUSHPLUS Notification Support:** Receive notifications about IP updates and potential issues via PUSHPLUS.
*   **Simple API Access:** Easily retrieve the top Cloudflare IPs for use in other applications.

## Configuration via GitHub Actions:

Configure your DNS settings using GitHub Actions secrets and variables. The available integrations are:

*   **DNSPOD:**
    *   `DOMAIN`: Your domain name (e.g., `164746.xyz`)
    *   `SUB_DOMAIN`: Your subdomain (e.g., `dns`)
    *   `SECRETID`: Your DNSPOD Secret ID
    *   `SECRETKEY`: Your DNSPOD Secret Key
    *   `PUSHPLUS_TOKEN`: (Optional) Your PUSHPLUS token for notifications.

*   **DNSCF:**
    *   `CF_API_TOKEN`: Your Cloudflare API Token.
    *   `CF_ZONE_ID`: Your Cloudflare Zone ID.
    *   `CF_DNS_NAME`:  Your DNS record name (e.g., `dns.164746.xyz`).
    *   `PUSHPLUS_TOKEN`: (Optional) Your PUSHPLUS token for notifications.

## API Endpoint:

Get the top Cloudflare IPs:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

### API Response Example

```
104.16.204.6,104.18.103.125
```

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Supporting the Project

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")