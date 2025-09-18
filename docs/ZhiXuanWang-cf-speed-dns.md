# Maximize Your Cloudflare CDN Speed with cf-speed-dns

**Tired of slow website loading times?** **cf-speed-dns** automatically finds and updates your Cloudflare DNS settings with the fastest IP addresses, providing optimal performance and speed for your website.  [See the original repository here](https://github.com/ZhiXuanWang/cf-speed-dns).

## Key Features of cf-speed-dns:

*   **Real-time Cloudflare IP Optimization:** Automatically identifies and updates your DNS records with the fastest Cloudflare CDN IP addresses, ensuring the best possible website performance.
*   **Optimized IP Lists:** Access pre-generated lists of optimized Cloudflare IPs:
    *   [Top IPs](https://ip.164746.xyz/ipTop.html) (Default)
    *   [Top 10 IPs](https://ip.164746.xyz/ipTop10.html)
*   **Dynamic DNS Updates:**  Integrates with popular DNS providers for seamless updates.
    *   DNSPOD integration via GitHub Actions.
    *   DNSCF integration via GitHub Actions.
*   **Customizable Notifications:** Receive notifications via PUSHPLUS to stay informed about updates.  Configure your PUSHPLUS token: [https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html)
*   **Easy Integration:**  Uses GitHub Actions for automated DNS updates, making setup straightforward.

## How to Use the API:

Get the top optimized IP addresses using a simple `curl` request:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

The API returns a comma-separated list of the fastest IPs, for example:

```
104.16.204.6,104.18.103.125
```

## Configuration (GitHub Actions):

Configure the following secrets and variables in your GitHub Actions workflow for DNS updates:

*   **DNSPOD:**
    *   `DOMAIN`: Your domain name (e.g., `164746.xyz`)
    *   `SUB_DOMAIN`: Your subdomain (e.g., `dns`)
    *   `SECRETID`: Your DNSPOD Secret ID.
    *   `SECRETKEY`: Your DNSPOD Secret Key.
    *   `PUSHPLUS_TOKEN`: Your PUSHPLUS Token.
*   **DNSCF:**
    *   `CF_API_TOKEN`: Your Cloudflare API Token.
    *   `CF_ZONE_ID`: Your Cloudflare Zone ID.
    *   `CF_DNS_NAME`:  Your DNS record name (e.g., `dns.164746.xyz`)
    *   `PUSHPLUS_TOKEN`: Your PUSHPLUS Token.

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")