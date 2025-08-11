# Find the Fastest Cloudflare IPs with cf-speed-dns

**Optimize your Cloudflare performance by automatically finding and using the fastest Cloudflare IP addresses.** This tool, [cf-speed-dns](https://github.com/ZhiXuanWang/cf-speed-dns), helps you achieve optimal speed and reliability.

## Key Features

*   **Real-time Cloudflare IP Optimization:** Continuously identifies and updates the best-performing Cloudflare IP addresses.
*   **Optimized IP Lists:** Provides access to prioritized lists of fast IPs via endpoints:
    *   [https://ip.164746.xyz](https://ip.164746.xyz) (Live Optimized IP List)
    *   [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) (Top IPs - Default)
    *   [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html) (Top 10 IPs)
*   **Automated DNS Updates:**  Integrates with DNSPOD and DNSCF for automatic DNS record updates using GitHub Actions.
*   **Notification Support:**  Sends notifications via PUSHPLUS.
*   **Easy Integration:** Simple configuration using GitHub Actions secrets for seamless setup.

##  How it Works

cf-speed-dns uses CloudflareSpeedTest to identify the fastest Cloudflare IPs. These IPs are then made available via the provided API endpoints. The tool can be configured to automatically update your DNS records with these optimized IPs.

## API Endpoint Example

You can retrieve the current top IPs using a simple `curl` command:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

This will return a comma-separated list of the top-performing Cloudflare IP addresses, for example:

```
104.16.204.6,104.18.103.125
```

## DNS Configuration via GitHub Actions

To set up automatic DNS updates, fork this project and configure the following secrets in your GitHub Actions:

*   **DNSPOD Integration:**
    *   `DOMAIN`: Your domain (e.g., `164746.xyz`)
    *   `SUB_DOMAIN`: The subdomain to update (e.g., `dns`)
    *   `SECRETID`: Your DNSPOD Secret ID
    *   `SECRETKEY`: Your DNSPOD Secret Key
    *   `PUSHPLUS_TOKEN`: Your PUSHPLUS notification token (optional)
*   **DNSCF Integration:**
    *   `CF_API_TOKEN`: Your Cloudflare API Token
    *   `CF_ZONE_ID`: Your Cloudflare Zone ID
    *   `CF_DNS_NAME`: The DNS record to update (e.g., `dns.164746.xyz`)
    *   `PUSHPLUS_TOKEN`: Your PUSHPLUS notification token (optional)

## Acknowledgements

This project leverages the work of the following:

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Support & Sponsorship

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")