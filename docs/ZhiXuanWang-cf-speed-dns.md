# Optimize Your Cloudflare CDN Performance with cf-speed-dns

**Quickly find and utilize the fastest Cloudflare IP addresses to optimize your website's performance.** This tool leverages CloudflareSpeedTest to identify and update the best-performing IP addresses, ensuring your content is delivered to your users as quickly as possible.

[**Visit the original repository for more information and to get started!**](https://github.com/ZhiXuanWang/cf-speed-dns)

## Key Features

*   **Real-Time IP Selection:** Automatically identifies and updates the fastest Cloudflare IP addresses every 5 minutes.
*   **Optimized IP Lists:** Access pre-configured lists of optimized IPs via these endpoints:
    *   [https://ip.164746.xyz](https://ip.164746.xyz) (General list)
    *   [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) (Top IPs - default)
    *   [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html) (Top 10 IPs)
*   **DNSPOD Integration:** Seamlessly integrates with DNSPOD for dynamic DNS updates. (Requires configuration of secrets and variables: DOMAIN, SUB\_DOMAIN, SECRETID, SECRETKEY, PUSHPLUS\_TOKEN)
*   **DNSCF Integration:** Integrates with DNSCF for dynamic DNS updates. (Requires configuration of secrets and variables: CF\_API\_TOKEN, CF\_ZONE\_ID, CF\_DNS\_NAME, PUSHPLUS\_TOKEN)
*   **Push Notification Support:** Receive notifications using Pushplus ([https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html)) to stay informed about IP updates.

## API Endpoint

Get the top performing IPs:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

## API Response (Example)

```javascript
104.16.204.6,104.18.103.125
```

## Acknowledgements

Special thanks to:

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Supporting Open Source

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")