# Optimize Your Cloudflare CDN Performance with cf-speed-dns

**Quickly identify and utilize the fastest Cloudflare IPs for optimal website speed and performance.**  ([View the original repository](https://github.com/ZhiXuanWang/cf-speed-dns))

## Key Features:

*   **Real-time Cloudflare IP Optimization:**  Leverages CloudflareSpeedTest to identify and provide the lowest latency, fastest Cloudflare IPs.
*   **Optimized IP Lists:** Access pre-sorted IP lists for immediate use:
    *   Top IPs:  [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) (default)
    *   Top 10 IPs: [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
    *   Full List: [https://ip.164746.xyz](https://ip.164746.xyz)
*   **Automated DNS Updates:** Integrated with DNSPOD and DNSCF for automatic DNS record updates with the optimized IPs, simplifying the process of switching IPs to the fastest ones.  Configure using GitHub Actions.
*   **Notification System:**  Integrates with PUSHPLUS for real-time notifications of IP updates.
*   **Easy to Use:**  Simple API for retrieving the optimized IP list:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

*   **API Response:** Returns a comma-separated list of the fastest IPs.
```json
104.16.204.6,104.18.103.125
```

## Configuration (GitHub Actions)

To use the DNS update features (DNSPOD or DNSCF), you'll need to fork this project and configure the necessary secrets in your GitHub repository:

### DNSPOD Configuration:

1.  **Fork the repository.**
2.  In your forked repository, go to "Settings" -> "Secrets and variables" -> "Actions".
3.  Add the following secrets:
    *   `DOMAIN`: Your domain name (e.g., `164746.xyz`)
    *   `SUB_DOMAIN`: Your subdomain (e.g., `dns`)
    *   `SECRETID`: Your DNSPOD Secret ID
    *   `SECRETKEY`: Your DNSPOD Secret Key
    *   `PUSHPLUS_TOKEN` (Optional): Your PUSHPLUS token for notifications.

### DNSCF Configuration:

1.  **Fork the repository.**
2.  In your forked repository, go to "Settings" -> "Secrets and variables" -> "Actions".
3.  Add the following secrets:
    *   `CF_API_TOKEN`: Your Cloudflare API Token.
    *   `CF_ZONE_ID`: Your Cloudflare Zone ID.
    *   `CF_DNS_NAME`: Your DNS record name (e.g., `dns.164746.xyz`).
    *   `PUSHPLUS_TOKEN` (Optional): Your PUSHPLUS token for notifications.

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Advertisement

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")