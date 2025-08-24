# Optimize Your Cloudflare CDN with cf-speed-dns: Find the Fastest IPs!

Tired of slow Cloudflare performance? **cf-speed-dns helps you automatically identify and use the quickest Cloudflare IPs for optimal speed and performance.**  This project leverages CloudflareSpeedTest to find the best IPs and then automates DNS updates, ensuring you're always using the fastest connection.  Learn more and contribute at the [original repository](https://github.com/ZhiXuanWang/cf-speed-dns).

## Key Features:

*   **Real-time Cloudflare IP Optimization:** Identifies and provides a list of the fastest Cloudflare IPs.
*   **Multiple IP Lists:** Access optimized IP lists through various interfaces:
    *   [Top IPs Interface](https://ip.164746.xyz/ipTop.html) (Default)
    *   [Top 10 IPs Interface](https://ip.164746.xyz/ipTop10.html)
*   **Automated DNS Updates (via GitHub Actions):**
    *   **DNSPOD Integration:**  Automatically updates your DNS records with the fastest IPs. Requires configuration of secrets: `DOMAIN`, `SUB_DOMAIN`, `SECRETID`, `SECRETKEY`, and `PUSHPLUS_TOKEN`.
    *   **DNSCF Integration:**  Updates your DNS records via Cloudflare's API.  Requires configuration of secrets: `CF_API_TOKEN`, `CF_ZONE_ID`, `CF_DNS_NAME`, and `PUSHPLUS_TOKEN`.
*   **PUSHPLUS Notifications:** Receive notifications about DNS updates. ([PUSHPLUS](https://www.pushplus.plus/push1.html))

## How it Works

The project uses CloudflareSpeedTest to measure the latency of different Cloudflare IPs. It then provides an interface with the best performing IPs which can be integrated with Github actions to automate the updating of DNS records.

## API Endpoint

The `ipTop.html` endpoint returns a comma-separated list of the fastest Cloudflare IPs.

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

## API Response Example

```
104.16.204.6,104.18.103.125
```

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Supported By
[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")