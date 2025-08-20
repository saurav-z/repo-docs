# Find the Fastest Cloudflare IPs with cf-speed-dns

**Optimize your Cloudflare experience with cf-speed-dns, instantly identifying and utilizing the quickest Cloudflare IPs for superior speed and performance.** (Original Repository: [ZhiXuanWang/cf-speed-dns](https://github.com/ZhiXuanWang/cf-speed-dns))

## Key Features of cf-speed-dns

*   **Real-time Cloudflare IP Optimization:** Identifies and lists the fastest Cloudflare IPs every 5 minutes.
*   **Up-to-the-Minute IP Lists:** Access updated lists via a web interface:  [https://ip.164746.xyz](https://ip.164746.xyz)
*   **Top IP Selection:** Retrieve top-performing IPs:
    *   Top Interface (Default): [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html)
    *   Top 10 Interface: [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)
*   **Automated DNS Updates:** Integrate with DNSPOD and DNSCF for automatic domain resolution updates.
    *   **DNSPOD Integration:** Configure via GitHub Actions, using DOMAIN, SUB\_DOMAIN, SECRETID, SECRETKEY, and PUSHPLUS\_TOKEN secrets.
    *   **DNSCF Integration:** Configure via GitHub Actions, using CF\_API\_TOKEN, CF\_ZONE\_ID, CF\_DNS\_NAME, and PUSHPLUS\_TOKEN secrets.
*   **Push Notifications:** Receive notifications via PUSHPLUS.  [https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html)

## API Access

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

## API Response Example

```javascript
104.16.204.6,104.18.103.125
```

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Powered by

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")