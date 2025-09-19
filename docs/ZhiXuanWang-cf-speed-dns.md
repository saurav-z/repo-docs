# Cloudflare Speed DNS: Find and Use the Fastest Cloudflare IPs

**Optimize your Cloudflare performance with cf-speed-dns, automatically finding and updating your DNS settings with the fastest Cloudflare IP addresses.**

This tool helps you identify the optimal Cloudflare IP addresses for your needs, improving website speed and performance.  You can find the original repository here: [https://github.com/ZhiXuanWang/cf-speed-dns](https://github.com/ZhiXuanWang/cf-speed-dns).

## Key Features

*   **Real-time Cloudflare IP Selection:** Automatically identifies and updates the fastest Cloudflare IPs.
*   **Optimized IP Lists:** Provides access to lists of the best-performing IPs:
    *   [https://ip.164746.xyz](https://ip.164746.xyz) - Real-time updated list of optimized IPs.
    *   [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) - Top interface (default).
    *   [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html) - Top 10 interface.
*   **DNS Record Updates:** Supports automatic DNS record updates via DNSPOD and DNSCF.  This is done by forking this project and configuring GitHub Actions.
    *   **DNSPOD Integration:** Configure with DOMAIN, SUB\_DOMAIN, SECRETID, SECRETKEY, and (optional) PUSHPLUS\_TOKEN for notifications.
    *   **DNSCF Integration:** Configure with CF\_API\_TOKEN, CF\_ZONE\_ID, CF\_DNS\_NAME, and (optional) PUSHPLUS\_TOKEN for notifications.
*   **PUSHPLUS Notifications:** Integrates with PUSHPLUS for customizable notifications ([https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html)).

## API Endpoint

```javascript
curl 'https://ip.164746.xyz/ipTop.html'
```

## API Response Example

```javascript
104.16.204.6,104.18.103.125
```

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Powered By
[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")