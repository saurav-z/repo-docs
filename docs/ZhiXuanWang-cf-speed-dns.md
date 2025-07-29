# Optimize Your Cloudflare CDN Speed with cf-speed-dns

**Tired of slow website loading speeds?** This project, **cf-speed-dns**, automatically identifies and utilizes the fastest Cloudflare CDN IPs for optimal performance. Find the original project at [https://github.com/ZhiXuanWang/cf-speed-dns](https://github.com/ZhiXuanWang/cf-speed-dns).

## Key Features:

*   **Real-time Optimized IP Selection:**  Leverages CloudflareSpeedTest to identify and provide the lowest latency Cloudflare CDN IPs every 5 minutes.
*   **Multiple IP List Endpoints:** Access optimized IP lists via various endpoints:
    *   [https://ip.164746.xyz](https://ip.164746.xyz) (Main list)
    *   [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) (Top IPs - Default)
    *   [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html) (Top 10 IPs)
*   **Automated DNS Updates:** Supports real-time domain resolution updates through:
    *   **DNSPOD Integration:** Configure with your DOMAIN, SUB\_DOMAIN, SECRETID, SECRETKEY, and PUSHPLUS\_TOKEN.
    *   **DNSCF Integration:** Configure with your CF\_API\_TOKEN, CF\_ZONE\_ID, CF\_DNS\_NAME, and PUSHPLUS\_TOKEN.
*   **PUSHPLUS Notifications:**  Receive notifications about IP updates through PUSHPLUS. ([https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html))

## API Endpoint Example:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

## API Response Example:

```
104.16.204.6,104.18.103.125
```

## Acknowledgements

Special thanks to:

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)