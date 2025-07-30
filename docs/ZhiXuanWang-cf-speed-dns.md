# Optimize Your Cloudflare CDN with cf-speed-dns: Find the Fastest IPs!

**Tired of slow website loading times?** [cf-speed-dns](https://github.com/ZhiXuanWang/cf-speed-dns) automatically finds and updates the fastest Cloudflare CDN IPs for optimal performance.

## Key Features

*   **Automated IP Selection:**  Identifies and pushes the best-performing Cloudflare IPs every 5 minutes.
*   **Real-time Updates:**  Keeps your IP list fresh with a constantly updated list.
*   **Flexible Top IP Lists:** Provides access to a list of top IPs ([https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html)), and a list of top 10 IPs ([https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html)).
*   **DNSPOD Integration:** Easily integrates with DNSPOD for real-time domain name resolution updates. Requires configuration of secrets and variables in GitHub Actions (DOMAIN, SUB\_DOMAIN, SECRETID, SECRETKEY, PUSHPLUS\_TOKEN).
*   **DNSCF Integration:** Enables real-time domain name resolution using Cloudflare DNS. Configure with CF\_API\_TOKEN, CF\_ZONE\_ID, CF\_DNS\_NAME, and PUSHPLUS\_TOKEN in GitHub Actions.
*   **PUSHPLUS Notifications:**  Receive notifications when the IP lists are updated ([https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html)).

## API Usage

Get the top-performing Cloudflare IPs:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

**Response Example:**

```
104.16.204.6,104.18.103.125
```

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Powered By

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")