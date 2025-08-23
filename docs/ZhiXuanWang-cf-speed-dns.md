# Optimize Your Cloudflare CDN Performance with cf-speed-dns

**Tired of slow website loading times?**  `cf-speed-dns` automatically finds and updates the fastest Cloudflare CDN IP addresses, ensuring optimal website performance and a superior user experience.  [View the original repository here](https://github.com/ZhiXuanWang/cf-speed-dns).

## Key Features:

*   **Real-time Cloudflare IP Optimization:**  Continuously updates a list of the fastest Cloudflare CDN IPs.
*   **Pre-Built IP Lists:** Access optimized IP lists via these convenient endpoints:
    *   **Top IPs:**  [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) (Default - provides the fastest IPs)
    *   **Top 10 IPs:**  [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **Automated DNS Updates:**  Integrates with popular DNS providers to automatically update your DNS records with the optimal Cloudflare IPs.  Supports:
    *   DNSPOD
    *   DNSCF
*   **Customizable Notifications:**  Receive instant updates via PUSHPLUS notification service.
*   **Easy Implementation:**  Set up automated updates via GitHub Actions.
*   **Simple API Access:** Get the current top IPs with a simple cURL command.

## Getting Started:

### API Request:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

### API Response (Example):

```
104.16.204.6,104.18.103.125
```

##  Setting Up DNS Updates (via GitHub Actions):

1.  **Fork this repository.**
2.  **Configure GitHub Actions Secrets and Variables:**

    *   **DNSPOD:** Add the following secrets/variables: `DOMAIN`, `SUB_DOMAIN`, `SECRETID`, `SECRETKEY`, `PUSHPLUS_TOKEN`
    *   **DNSCF:** Add the following secrets/variables: `CF_API_TOKEN`, `CF_ZONE_ID`, `CF_DNS_NAME`, `PUSHPLUS_TOKEN`

3.  **Enable DNS Updates:**  The included GitHub Actions workflows will automatically update your DNS records based on the fetched Cloudflare IP data.

##  Notifications:

*   Set up notifications using PUSHPLUS: [https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html)

##  Acknowledgments:

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

##  Sponsored by:

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")