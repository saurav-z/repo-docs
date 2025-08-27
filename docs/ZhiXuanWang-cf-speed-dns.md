# cf-speed-dns: Find & Deploy the Fastest Cloudflare IPs

**Optimize your website's performance by automatically identifying and deploying the quickest Cloudflare CDN IPs with cf-speed-dns.**

Learn more and contribute to the original project on GitHub: [https://github.com/ZhiXuanWang/cf-speed-dns](https://github.com/ZhiXuanWang/cf-speed-dns)

## Key Features

*   **Real-time Cloudflare IP Optimization:** Identifies and updates the fastest Cloudflare IPs every 5 minutes.
*   **Optimized IP Lists:** Provides access to optimized IP lists, including:
    *   Top IPs: [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) (default)
    *   Top 10 IPs: [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **Automated DNS Updates:** Seamless integration with DNSPOD and DNSCF for real-time domain name resolution updates.
*   **Notification Integration:** Supports PUSHPLUS for instant notifications.
*   **Easy Deployment:** Configure through GitHub Actions secrets and variables.

## How it Works

cf-speed-dns leverages CloudflareSpeedTest to find the optimal Cloudflare CDN IPs. It then provides these IPs through accessible interfaces and automates the updating of your DNS records, leading to faster website loading speeds and improved user experience.

## API Endpoint Example

Retrieve the top Cloudflare IPs:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

Example Response:

```
104.16.204.6,104.18.103.125
```

## Configuration

### DNSPOD Integration
1.  Fork this project.
2.  In your GitHub repository, navigate to **Settings > Secrets and variables > Actions**.
3.  Add the following secrets:
    *   `DOMAIN`: Your domain name (e.g., `164746.xyz`).
    *   `SUB_DOMAIN`: Your subdomain (e.g., `dns`).
    *   `SECRETID`: Your DNSPOD Secret ID.
    *   `SECRETKEY`: Your DNSPOD Secret Key.
    *   `PUSHPLUS_TOKEN`: (Optional) Your PUSHPLUS token for notifications.

### DNSCF Integration
1.  Fork this project.
2.  In your GitHub repository, navigate to **Settings > Secrets and variables > Actions**.
3.  Add the following secrets:
    *   `CF_API_TOKEN`: Your Cloudflare API token.
    *   `CF_ZONE_ID`: Your Cloudflare Zone ID.
    *   `CF_DNS_NAME`: The DNS record name (e.g., `dns.164746.xyz`).
    *   `PUSHPLUS_TOKEN`: (Optional) Your PUSHPLUS token for notifications.

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Sponsorship

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")