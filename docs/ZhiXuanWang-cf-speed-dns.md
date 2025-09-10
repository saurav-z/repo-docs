# Optimize Your Cloudflare Performance with cf-speed-dns

**Tired of slow Cloudflare speeds?**  [cf-speed-dns](https://github.com/ZhiXuanWang/cf-speed-dns) automatically finds and updates your DNS records with the fastest Cloudflare IP addresses, ensuring optimal performance for your website.

## Key Features:

*   **Real-time Optimized IPs:**  Continuously identifies and updates your DNS with the best-performing Cloudflare IP addresses.
*   **Multiple IP Selection Options:** Choose from a variety of IP lists, including top IPs and top 10 IPs.
*   **DNSPOD Integration:**  Automatically updates your DNS records on DNSPOD with the selected optimal IPs. Configuration via GitHub Actions.
*   **DNSCF Integration:**  Automatically updates your DNS records on Cloudflare with the selected optimal IPs. Configuration via GitHub Actions.
*   **PUSHPLUS Notifications:** Get notified about IP updates and other important events.
*   **Easy-to-Use API:** Access the latest optimized IP addresses through a simple API endpoint.

## API Endpoints:

*   **Top IPs:** `https://ip.164746.xyz/ipTop.html` (Returns the top performing IPs)
*   **Top 10 IPs:** `https://ip.164746.xyz/ipTop10.html`

**Example API Request:**

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

**Example API Response:**

```
104.16.204.6,104.18.103.125
```

## Configuration (DNSPOD & DNSCF - via GitHub Actions)

*   **Fork this project** to get started.
*   Configure secrets and variables in your GitHub Actions settings:
    *   **DNSPOD:** `DOMAIN`, `SUB_DOMAIN`, `SECRETID`, `SECRETKEY`, `PUSHPLUS_TOKEN`
    *   **DNSCF:** `CF_API_TOKEN`, `CF_ZONE_ID`, `CF_DNS_NAME`, `PUSHPLUS_TOKEN`

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Powered by

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")