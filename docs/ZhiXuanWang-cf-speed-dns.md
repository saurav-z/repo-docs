# Find the Fastest Cloudflare IPs with cf-speed-dns

This tool helps you find and automatically update your Cloudflare DNS records with the fastest IPs, optimizing your website's speed and performance. ([Original Repo](https://github.com/ZhiXuanWang/cf-speed-dns))

## Key Features

*   **Real-time Cloudflare IP Optimization:** Automatically identifies and updates your Cloudflare DNS records with the fastest-performing IPs.
*   **Optimized IP Lists:** Provides pre-generated lists of optimized Cloudflare IPs, updated every 5 minutes, for immediate use.
*   **Multiple Interface Options:** Offers several interface options to choose from, including:
    *   [https://ip.164746.xyz](https://ip.164746.xyz) (Main list)
    *   [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) (Top IPs - Default)
    *   [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html) (Top 10 IPs)
*   **DNSPOD & DNSCF Integration:** Supports automated DNS record updates through DNSPOD and DNSCF, using GitHub Actions.
*   **Notification Support:** Integrates with PUSHPLUS for real-time notifications on updates.
*   **Easy Setup:** Configuration using GitHub Actions secrets and variables.

## How it Works

cf-speed-dns uses CloudflareSpeedTest to determine the fastest Cloudflare IPs and automatically updates your DNS records. The following configuration is needed.

### **DNSPOD Configuration**

1.  **Fork this project.**
2.  **Configure GitHub Actions secrets and variables:**
    *   `DOMAIN`: Your domain (e.g., `164746.xyz`)
    *   `SUB_DOMAIN`: Your subdomain (e.g., `dns`)
    *   `SECRETID`: Your DNSPOD Secret ID
    *   `SECRETKEY`: Your DNSPOD Secret Key
    *   `PUSHPLUS_TOKEN` (Optional): Your Pushplus token for notifications.

### **DNSCF Configuration**

1.  **Fork this project.**
2.  **Configure GitHub Actions secrets and variables:**
    *   `CF_API_TOKEN`: Your Cloudflare API Token
    *   `CF_ZONE_ID`: Your Cloudflare Zone ID
    *   `CF_DNS_NAME`: Your DNS name (e.g., `dns.164746.xyz`)
    *   `PUSHPLUS_TOKEN` (Optional): Your Pushplus token for notifications.

## API Endpoint

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

## Advertisement

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")