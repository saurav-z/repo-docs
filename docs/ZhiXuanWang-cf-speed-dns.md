# cf-speed-dns: Optimize Your Cloudflare CDN with Speed-Optimized IPs

**Instantly boost your website's performance by leveraging the fastest Cloudflare CDN IPs with cf-speed-dns.**

This tool automatically identifies and updates your Cloudflare DNS records with the most performant IPs, ensuring optimal speed and reliability for your visitors. [View the original repository on GitHub](https://github.com/ZhiXuanWang/cf-speed-dns).

## Key Features:

*   **Real-time Cloudflare IP Optimization:**  Continuously selects and provides the fastest Cloudflare IP addresses.
*   **Optimized IP Lists:** Access pre-built lists of top-performing IPs via:
    *   Top Interface:  `https://ip.164746.xyz/ipTop.html` (default)
    *   Top 10 Interface: `https://ip.164746.xyz/ipTop10.html`
*   **Automated DNS Updates:**  Integrates with DNSPOD and DNSCF to automatically update your DNS records with the selected optimal IPs. Configure this via GitHub Actions.
*   **Push Notifications:**  Receive notifications about IP updates via PUSHPLUS.
*   **Simple API:**  Easily retrieve the top performing IPs using a simple cURL command (see below).

## Integration and Configuration

### DNSPOD Integration:

1.  Fork this repository.
2.  Configure the following secrets and variables in your GitHub Actions settings:
    *   `DOMAIN`: Your domain name (e.g., `164746.xyz`)
    *   `SUB_DOMAIN`: Your subdomain (e.g., `dns`)
    *   `SECRETID`: Your DNSPOD Secret ID
    *   `SECRETKEY`: Your DNSPOD Secret Key
    *   `PUSHPLUS_TOKEN`: Your PUSHPLUS token (optional, for notifications)

### DNSCF Integration:

1.  Fork this repository.
2.  Configure the following secrets and variables in your GitHub Actions settings:
    *   `CF_API_TOKEN`: Your Cloudflare API Token
    *   `CF_ZONE_ID`: Your Cloudflare Zone ID
    *   `CF_DNS_NAME`: The DNS record you want to update (e.g., `dns.164746.xyz`)
    *   `PUSHPLUS_TOKEN`: Your PUSHPLUS token (optional, for notifications)

## API Usage

Get the top-performing Cloudflare IPs:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

### API Response Example:

```
104.16.204.6,104.18.103.125
```

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Powered by

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")