# Optimize Your Cloudflare CDN Performance with cf-speed-dns

**Tired of slow website loading speeds?**  This tool helps you find and automatically use the fastest Cloudflare IP addresses for optimal performance.  (See the original repository [here](https://github.com/ZhiXuanWang/cf-speed-dns).)

## Key Features:

*   **Real-time Cloudflare IP Optimization:** Automatically identifies and utilizes the fastest Cloudflare IPs.
*   **Optimized IP Lists:**  Provides updated lists of the best-performing IPs, including top-performing IPs and top 10 lists.
*   **DNSPOD Integration:**  Integrates with DNSPOD to automatically update your DNS records with the optimized IP addresses. (Requires configuration, see below)
*   **DNSCF Integration:**  Integrates with Cloudflare to automatically update your DNS records with the optimized IP addresses. (Requires configuration, see below)
*   **PUSHPLUS Notifications:**  Receive notifications via PUSHPLUS to stay informed about IP updates and performance.
*   **Easy-to-Use API:** Simple API endpoints for accessing the top-performing IPs.

## Configuration for DNS Updates (Requires Forking the Project):

To leverage the automatic DNS update features, you'll need to fork this project and configure the following secrets in your GitHub Actions settings:

*   **DNSPOD Integration (Choose one):**
    *   `DOMAIN`: Your domain name (e.g., `164746.xyz`).
    *   `SUB_DOMAIN`: Your subdomain (e.g., `dns`).
    *   `SECRETID`: Your DNSPOD Secret ID.
    *   `SECRETKEY`: Your DNSPOD Secret Key.
    *   `PUSHPLUS_TOKEN`: Your PUSHPLUS notification token.
*   **Cloudflare DNS Integration (Choose one):**
    *   `CF_API_TOKEN`: Your Cloudflare API Token.
    *   `CF_ZONE_ID`: Your Cloudflare Zone ID.
    *   `CF_DNS_NAME`: Your Cloudflare DNS record name (e.g., `dns.164746.xyz`).
    *   `PUSHPLUS_TOKEN`: Your PUSHPLUS notification token.

## API Endpoints:

*   **Top IPs (Default):** `https://ip.164746.xyz/ipTop.html`
*   **Top 10 IPs:** `https://ip.164746.xyz/ipTop10.html`

**Example API Request:**

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

**Example API Response:**

```
104.16.204.6,104.18.103.125
```

## Acknowledgements

This project utilizes code and inspiration from:

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Advertising

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")