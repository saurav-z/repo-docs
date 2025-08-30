# Get the Fastest Cloudflare IP Addresses with cf-speed-dns

Tired of slow website loading times? **cf-speed-dns automatically finds and updates the fastest Cloudflare IP addresses for optimal performance.**

This project, a fork of [CloudflareSpeedTest](https://github.com/ZhiXuanWang/cf-speed-dns), helps you identify and utilize the quickest Cloudflare CDN IPs. It provides tools for both finding optimal IPs and automatically updating your DNS records with them.

## Key Features:

*   **Real-time Cloudflare IP Optimization:** Scans and identifies the fastest Cloudflare IP addresses.
*   **Optimized IP Lists:** Access pre-built lists of top-performing IPs:
    *   [https://ip.164746.xyz](https://ip.164746.xyz) (Live list of optimal IPs)
    *   [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) (Top IP - Default)
    *   [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html) (Top 10 IPs)
*   **Automated DNS Updates:** Supports automated updates to DNS records via:
    *   DNSPOD integration.
    *   DNSCF integration.
*   **PUSHPLUS Notifications:** Receive notifications about IP updates and other events.
*   **Easy to Use:** Simple setup through GitHub Actions, using secrets and variables.

##  How it Works:

The project uses the following API endpoint to return a list of optimal IPs:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

The API returns a comma-separated list of IP addresses.

```
104.16.204.6,104.18.103.125
```

## Setup and Configuration:

(Detailed setup instructions would go here, including instructions for setting up GitHub Actions and configuring secrets.  This section is missing from the original README.)

## Acknowledgements:

This project leverages the work of:

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Original Repository:

Find the original project here: [https://github.com/ZhiXuanWang/cf-speed-dns](https://github.com/ZhiXuanWang/cf-speed-dns)

##  (Optional) Advertisement:

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")