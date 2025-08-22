# Optimize Your Cloudflare CDN with cf-speed-dns

**Instantly boost your Cloudflare CDN performance with `cf-speed-dns`, automatically selecting the fastest IPs for optimal speed and latency.**  This project provides a streamlined way to identify and utilize the best-performing Cloudflare IPs, ensuring your website or application runs at peak efficiency.  Get started today and experience a faster, more responsive online presence.  For the original source code, visit the [cf-speed-dns repository on GitHub](https://github.com/ZhiXuanWang/cf-speed-dns).

## Key Features of cf-speed-dns:

*   **Cloudflare IP Optimization:**  Dynamically selects and provides the best Cloudflare IP addresses based on speed and latency.
*   **Real-Time IP List Updates:**  Access a constantly updated list of optimized IPs via the following interfaces:
    *   Main List:  [https://ip.164746.xyz](https://ip.164746.xyz)
    *   Top IPs (Default): [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html)
    *   Top 10 IPs: [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **Automated DNS Updates:** Integrates with both DNSPOD and DNSCF for automated domain name resolution using the optimized IPs.
    *   **DNSPOD Integration:** Configure using Actions secrets and variables: `DOMAIN`, `SUB_DOMAIN`, `SECRETID`, `SECRETKEY`, and optionally `PUSHPLUS_TOKEN`.
    *   **DNSCF Integration:** Configure using Actions secrets and variables: `CF_API_TOKEN`, `CF_ZONE_ID`, `CF_DNS_NAME`, and optionally `PUSHPLUS_TOKEN`.
*   **PUSHPLUS Notification Support:** Receive notifications through PUSHPLUS for status updates.  Configure your `PUSHPLUS_TOKEN`.
*   **Easy-to-Use API Endpoint:**  Retrieve the top Cloudflare IPs with a simple `curl` command:
    ```bash
    curl 'https://ip.164746.xyz/ipTop.html'
    ```
*   **API Response Format:** The API returns a comma-separated list of the fastest IPs, such as:
    ```
    104.16.204.6,104.18.103.125
    ```

## Acknowledgements

This project utilizes and is inspired by the work of:

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Support

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")