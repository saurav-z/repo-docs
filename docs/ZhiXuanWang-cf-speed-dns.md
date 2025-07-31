# Maximize Your Cloudflare CDN Speed with cf-speed-dns

**Tired of slow website loading times?**  This tool automatically selects and updates the fastest Cloudflare CDN IPs, enhancing your website's performance.

For the original project, visit: [https://github.com/ZhiXuanWang/cf-speed-dns](https://github.com/ZhiXuanWang/cf-speed-dns)

## Key Features of cf-speed-dns:

*   **Dynamic IP Selection:**  Continuously identifies and uses the optimal Cloudflare IP addresses for the best speed and low latency.
*   **Real-time IP Updates:**  Leverages CloudflareSpeedTest to provide a live list of optimized IPs, viewable on the dedicated webpage: [https://ip.164746.xyz](https://ip.164746.xyz).
*   **Top IP Interface Options:** Access the fastest IPs through the top interface (default): [https://ip.164746.xyz/ipTop.html](https://ip.164746.xyz/ipTop.html) or the top 10 IPs via: [https://ip.164746.xyz/ipTop10.html](https://ip.164746.xyz/ipTop10.html).
*   **DNSPOD Integration:** Seamlessly integrates with DNSPOD for real-time domain resolution. Configure this feature by forking the project and adding your specific domain settings via Github Actions Secrets.
*   **DNSCF Integration:** Compatible with DNSCF for real-time domain resolution. Configure this feature by forking the project and adding your specific Cloudflare settings via Github Actions Secrets.
*   **PUSHPLUS Notification Support:**  Receive notifications via PUSHPLUS. Learn more here: [https://www.pushplus.plus/push1.html](https://www.pushplus.plus/push1.html)

## API Endpoint

Use the following command to fetch the top IPs:

```bash
curl 'https://ip.164746.xyz/ipTop.html'
```

## API Response Example

The API returns a comma-separated list of optimized IPs, for example:

```
104.16.204.6,104.18.103.125
```

## Acknowledgements

*   [XIU2](https://github.com/XIU2/CloudflareSpeedTest)
*   [ddgth](https://github.com/ddgth/cf2dns)

## Advertisement

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")