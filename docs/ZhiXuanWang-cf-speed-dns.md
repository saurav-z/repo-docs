# Optimize Your Cloudflare CDN with cf-speed-dns

**Tired of slow website loading times?** cf-speed-dns helps you find and automatically use the fastest Cloudflare IP addresses for optimal performance. ([View the original repository](https://github.com/ZhiXuanWang/cf-speed-dns))

## Key Features:

*   **Real-time Cloudflare IP Optimization:**  Find the quickest Cloudflare IP addresses every 5 minutes for improved website speed.
*   **Pre-built IP Lists:** Access pre-compiled lists of top-performing IPs via easy-to-use interfaces:
    *   **Top IPs:** [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) (default)
    *   **Top 10 IPs:** [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
    *   **Complete List:** [https://ip.164746.xyz](https://ip.164746.xyz)
*   **Automated DNS Updates:** Automatically updates your DNS records using DNSPOD or DNSCF:
    *   **DNSPOD Integration:** Easily configure with your DOMAIN, SUB\_DOMAIN, SECRETID, SECRETKEY, and PUSHPLUS\_TOKEN in GitHub Actions secrets.
    *   **DNSCF Integration:** Configure with your CF\_API\_TOKEN, CF\_ZONE\_ID, CF\_DNS\_NAME, and PUSHPLUS\_TOKEN in GitHub Actions secrets.
*   **PUSHPLUS Notifications:** Receive instant updates via Pushplus notifications.  [https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html)

## API Endpoint:

*   **Get the Top Cloudflare IPs:**
    ```bash
    curl 'https://ip.164746.xyz/ipTop.html'
    ```
*   **Example Response:**
    ```
    104.16.204.6,104.18.103.125
    ```

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Advertisement

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")