## Optimize Your Cloudflare CDN Speed with cf-speed-dns

**Tired of slow Cloudflare performance?**  cf-speed-dns automatically finds and updates the fastest Cloudflare IP addresses for optimal speed and performance.  This project leverages CloudflareSpeedTest to identify and utilize the best-performing IP addresses for your needs.  Check out the original project on [GitHub](https://github.com/ZhiXuanWang/cf-speed-dns).

### Key Features of cf-speed-dns:

*   **Real-time Optimized IP Selection:** Identifies and utilizes the fastest Cloudflare IP addresses.
*   **Updated IP Lists:** Provides access to real-time updated lists of optimized IPs.
    *   [Main List](https://ip.164746.xyz)
    *   [Top IPs](https://ip.164746.xyz/ipTop.html) (default)
    *   [Top 10 IPs](https://ip.164746.xyz/ipTop10.html)
*   **Automated DNS Updates:** Integrates with DNS providers like DNSPOD and DNSCF for automatic updates.  Simply fork the project and configure your preferred DNS provider.
    *   **DNSPOD Integration:** Requires configuration of `DOMAIN`, `SUB_DOMAIN`, `SECRETID`, `SECRETKEY`, and `PUSHPLUS_TOKEN` in your Action secrets.
    *   **DNSCF Integration:** Requires configuration of `CF_API_TOKEN`, `CF_ZONE_ID`, `CF_DNS_NAME`, and `PUSHPLUS_TOKEN` in your Action secrets.
*   **PUSHPLUS Notification Support:** Receive notifications via PUSHPLUS when IP updates occur. [PUSHPLUS](https://www.pushplus.plus/push1.html)
*   **Simple API Interface:**  Easily access the top performing IP addresses via a straightforward API.

### API Endpoint

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

### API Response Example

```
104.16.204.6,104.18.103.125
```

### Acknowledgements

Thanks to the developers of the following projects which inspired this one:

*   [XIU2/CloudflareSpeedTest](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth/cf2dns](https://github.com/ddgth/cf2dns)

### Advertisement

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")