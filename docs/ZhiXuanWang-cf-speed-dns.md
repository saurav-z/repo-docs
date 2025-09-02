# Find the Fastest Cloudflare IPs with cf-speed-dns

**Tired of slow website loading times?**  [cf-speed-dns](https://github.com/ZhiXuanWang/cf-speed-dns) automatically finds and updates the fastest Cloudflare IP addresses, optimizing your website's performance and speed.

## Key Features:

*   **Real-time Cloudflare IP Optimization:** Automatically identifies and provides the lowest latency, fastest Cloudflare IP addresses for optimal performance.
*   **Up-to-Date IP Lists:**  Provides up-to-date lists of optimized IPs, updated every 5 minutes.
*   **Multiple IP List Options:** Offers different lists, including a Top IPs list and a Top 10 IPs list, available via web interfaces.
*   **DNSPOD & DNSCF Integration:** Supports automatic DNS updates via DNSPOD and DNSCF, simplifying the process of updating your DNS records.
*   **PUSHPLUS Notification:** Integrates with PUSHPLUS for real-time notifications.
*   **Easy to Use:** Uses simple cURL commands to access the best performing IPs.

## Available IP List Interfaces:

*   **Main List:** [https://ip.164746.xyz](https://ip.164746.xyz)
*   **Top IPs (Default):** [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html)
*   **Top 10 IPs:** [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)

## Accessing the Optimized IPs (API Example):

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

## Example API Response:

```
104.16.204.6,104.18.103.125
```

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Powered by DartNode

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")