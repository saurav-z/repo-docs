# Optimize Your Cloudflare CDN Performance with cf-speed-dns

**Tired of slow website loading speeds?**  This project helps you find and automatically update to the fastest Cloudflare IP addresses, ensuring optimal performance for your website. Check out the original repository [here](https://github.com/ZhiXuanWang/cf-speed-dns).

## Key Features of cf-speed-dns

*   **Automatic IP Selection:**  Dynamically identifies and updates to the best-performing Cloudflare IP addresses every 5 minutes, using CloudflareSpeedTest data.
*   **Real-time IP Lists:** Provides access to real-time lists of optimized IPs.
    *   **Top IPs:**  [https://ip.164746.xyz](https://ip.164746.xyz) (Main Interface)
    *   **Top 10 IPs:** [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
    *   **Top Interface:** [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) (Default)
*   **DNS Record Updates:**  Integrates with DNS providers for automated domain resolution updates.  Supports:
    *   **DNSPOD:**  Uses GitHub Actions with configurable secrets for automated DNS updates. Configure DOMAIN, SUB\_DOMAIN, SECRETID, SECRETKEY, and PUSHPLUS\_TOKEN.
    *   **DNSCF:** Uses GitHub Actions with configurable secrets for automated DNS updates. Configure CF\_API\_TOKEN, CF\_ZONE\_ID, CF\_DNS\_NAME, and PUSHPLUS\_TOKEN.
*   **Push Notification Support:** Integrates with PUSHPLUS for real-time notifications: [https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html).

## API Usage

Get the top Cloudflare IPs with:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

**Example Response:**

```
104.16.204.6,104.18.103.125
```

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Support Our Sponsors

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")