# Speed Up Your Website in China with Optimized CNAME Records

Tired of slow website speeds in China? This project provides optimized CNAME records to improve the access speed and stability of your web pages hosted on platforms like Cloudflare, Vercel, and Netlify.  [Check out the original repository](https://github.com/xingpingcn/enhanced-FaaS-in-China) for the latest updates and more details.

[![GitHub Stars](https://img.shields.io/github/stars/xingpingcn/enhanced-FaaS-in-China?style=flat)](https://github.com/xingpingcn/enhanced-FaaS-in-China)
[![Last Commit](https://img.shields.io/github/last-commit/xingpingcn/enhanced-FaaS-in-China?display_timestamp=author&style=flat)](https://github.com/xingpingcn/enhanced-FaaS-in-China/commits/main)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fxingpingcn%2Fenhanced-FaaS-in-China&count_bg=%236167ED&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits-since-2024-7-8&edge_flat=false)](https://hits.seeyoufarm.com)

## Key Features

*   **Improved Speed and Stability:**  Enhance website access speed and reliability for users in China.
*   **Simplified Setup:** Easy configuration – simply change your CNAME record.
*   **Platform Support:**  Works with popular platforms like Vercel, Netlify, and Cloudflare.
*   **Optimized Routing:**  Leverages faster and more stable routes within China.

## Getting Started

To improve your website's performance in China, change your CNAME record to the following based on your hosting provider:

> [!NOTE]
>
> When changing your CNAME records, replace `app.domain.com` with the domain you want to improve performance on.

*   **Vercel:** `vercel-cname.xingpingcn.top`
*   **Netlify:** `netlify-cname.xingpingcn.top`
*   **Vercel & Netlify (Combined):** `verlify-cname.xingpingcn.top`

> [!IMPORTANT]
>
> Before using the combined CNAME records, it is recommended to change the CNAME to official CNAME records of the hosting provider first and make sure your SSL/TLS certificates have been generated. After that, you may set up the combined CNAME record.

*   **Cloudflare:** `cf-cname.xingpingcn.top`

> [!IMPORTANT]
>
> If your domain is hosted on Cloudflare, using this CNAME may result in 403 errors. It's recommended to host your domain on a non-Cloudflare platform and then delete your site from Cloudflare before using this service. If you're using Cloudflare's SaaS function, please follow the instructions in the [SaaS usage documentation](docs/how2use-SaaS-for-CF/how2use-SaaS-for-CF.md).

## Testing Your Speed

> [!WARNING]
>
> 1. When testing, always include the protocol (e.g., `https://`) and use multiple speed test websites as their results may vary.
> 1.  Avoid testing the service provider's domains directly as this can lead to false positives.

1.  Test after changing your CNAME record.
2.  Use speed test tools and input your domain information.
    ![how2test](img/how2test.png)

## Potential Issues & Troubleshooting

*   **ISP Restrictions:**  Occasional access issues may occur with certain ISPs (e.g., in Quanzhou).
*   **Testing Tools:**  Be mindful of potential inaccuracies with certain testing tools (e.g., itdog.cn).

## Why Use This?

1.  **Bypass Slow Routing:** Avoids potential routing through Southeast Asia, often used by official anycast networks, and leverages faster routes within China or through Europe/USA.
2.  **Enhanced Stability:** Offers more reliable access compared to official CNAMEs, which can sometimes have inconsistent performance.
3.  **Increased Reliability:** Reduces the risk of complete website downtime if one platform faces issues in China.

> [!NOTE]
>
> **Optimized performance for China**
>
> _Note: Results from the speed test might not be up-to-date_
>
> ![vercel-23点晚高峰](img/vercel-2024-9-29-23utc8.png)
> Vercel - Peak Evening Hours
> ![cf-23点晚高峰](img/cf-2024-9-29-23utc8.png)
> Cloudflare - Peak Evening Hours

## Speed Test Comparisons

> [!IMPORTANT]  
> _Note: Results from the speed test might not be up-to-date_

<details>
<summary>Click to view results</summary>

![cf-23点晚高峰](img/cf-2024-9-29-23utc8.png)
Cloudflare - Peak Evening Hours
![cf-22点晚高峰](img/cf-22.5utc8-2024-6-26.png)
Cloudflare - Peak Evening Hours
![cf-23点晚高峰-官方](img/cf-23utc8-auth.png)
Cloudflare - Peak Evening Hours - Official
![cf-22点晚高峰-官方](img/cf-22utc8-auth.png)
Cloudflare - Peak Evening Hours - Official
![vercel-23点晚高峰](img/vercel-2024-9-29-23utc8.png)
Vercel - Peak Evening Hours
![vercel-23点晚高峰-官方](img/vercel-23utc8-auth.png)
Vercel - Peak Evening Hours - Official
![netlify-23点晚高峰](img/netlify-23utc8.png)
Netlify - Peak Evening Hours
![netlify-23点晚高峰-官方](img/netlify-23utc8-auth.png)
Netlify - Peak Evening Hours - Official
![vercel中午](img/vercel-noon.png)
Vercel - Midday
![vercel中午-官方](img/vercel-noon-auth.png)
Vercel - Midday - Official
![netlify中午](img/netlify-noon.png)
Netlify - Midday
![netlify中午-官方](img/netlify-noon-auth.png)
Netlify - Midday - Official

</details>

## How It Works

The service monitors Cloudflare, Vercel, and Netlify IP addresses, tests their speeds, and automatically updates DNS records with the most stable and fastest IPs.  Updates typically occur every 40 minutes.

### IP Sources

<details>
<summary>Click to view</b></summary><br>

*   **Vercel**
    *   [Vercel IPs](https://gist.github.com/ChenYFan/fc2bd4ec1795766f2613b52ba123c0f8)
    *   Official `cname.vercel-dns.com.` A records.
*   **Netlify**
    *   A records provided by the official platform
*   **Cloudflare**
    *   Various Cloudflare paid users' IP addresses

*   **Default International IPs**

```json
{
  "VERCEL": "76.76.21.21",
  "NETLIFY": "75.2.60.5",
  "CF": "japan.com."
}
```

</details>

## Frequently Asked Questions

**Q: Why am I not seeing the performance improvements shown in the speed test comparisons?**<br>
A:

*   **Testing Platform:** Compare results across multiple speed test platforms.
*   **Domain Issues:** Free or cheap domain names (e.g., .xyz, .top) may be blocked due to carrier whitelisting.
*   **Origin Server:** Performance comparisons test static web pages hosted on edge servers.
*   **DNS Resolution:**
    1.  Cloudflare's DNS in mainland China can be slow and add extra DNS resolution delays. If possible, host your domain with a domestic authoritative DNS provider.
    2.  If your domain is hosted on Cloudflare or NS1.COM, please refer to [this issue](https://github.com/xingpingcn/enhanced-FaaS-in-China/issues/9#issuecomment-2379335329)
    3.  You can manually sync the IP addresses from the three JSON files in the root directory (updated every 40 minutes) into your A records.
    4.  **You may also encounter an issue with your Huawei Cloud DNS misidentifying users from mainland China as being from abroad, resulting in the default `japan.com` route.**

**Q: Why is my website inaccessible after configuring your CNAME?**<br>
A:

*   This is likely due to the `verlify-cname.xingpingcn.top` setting.  First, use the hosting provider's official CNAME and wait for the SSL certificate to generate. Then, reconfigure with the combined CNAME.  This is because this setting includes both platforms' IPs, and the platform may consider you don't own the domain when each access is to a different platform IP. But once the certificate is generated, it will be cached on the platform.
*   Netlify [supports uploading your own certificates](/netlify_cert/readme.md). If the problem persists, consider applying for an automatically renewing certificate.
*   If your website is deployed on Cloudflare and you are using `cf-cname.xingpingcn.top`, you may encounter 403 errors if your domain is hosted on Cloudflare. In this case, host your domain on a non-Cloudflare platform, **then delete your site from Cloudflare** before using the service.
*   If you are only experiencing issues with speed test platforms, refer to the [Testing Your Speed](#testing-your-speed) section.

<details>
<summary><b>Q: What are the differences between your service and the official CNAME?</b></summary>
<br>
A:

*   Official CNAMEs can be fast on average, but they lack stability and can sometimes become inaccessible in multiple provinces or experience high response times in specific provinces.
*   Our CNAMEs might not always offer the absolute fastest speeds, but they aim to keep the average response time under 1 second, with the longest response time controlled within 2 seconds and a minimum of provinces returning non-200 status codes.
</details>
<details>
<summary><b>Q: Why is the route parsing not accurate?</b></summary><br>

A: We are using the built-in routing of the authoritative DNS server, which can be prone to misinterpretation. If you want more precise route parsing, you can choose another DNS server, such as dnspod, and add the IP addresses from the [Netlify.json](https://raw.githubusercontent.com/xingpingcn/enhanced-FaaS-in-China/main/Netlify.json) or [Vercel.json](https://raw.githubusercontent.com/xingpingcn/enhanced-FaaS-in-China/main/Vercel.json) files to your A records. Alternatively, you can use `NS1.COM` as your authoritative DNS server and set up route parsing based on your ASN. Check out my [ASN list](https://github.com/xingpingcn/china-mainland-asn).

</details>

<details>
<summary><b>Q: Why are some routes (such as Telecom) resolving to the default IPs provided by the platform?</b></summary><br>

A: This is because the other IPs for that route are of poor quality, so we have temporarily stopped resolving them and are using the default IPs provided by the platform. You can deploy your website on both `vercel` and `netlify` and configure the CNAME to `verlify-cname.xingpingcn.top` to increase fault tolerance. The probability of both platforms failing simultaneously on the same route is much lower.

</details>

<details>
<summary><b>Q: Why is the route an empty list in the JSON file?</b></summary><br>

A: See the answer above.

</details>

## Customization

To add support for additional platforms (e.g., Render, Railway), you'll need to create a testing tool, a domain, and rewrite `crawler.py`. Create a `.py` file in the `platforms_to_test` folder, replicate the `run_sub()` method, and modify the configuration files in `config.py`.

## Star History

[![Stargazers over time](https://starchart.cc/xingpingcn/enhanced-FaaS-in-China.svg?background=%23FFFFFF&axis=%23333333&line=%23ff63db)](https://starchart.cc/xingpingcn/enhanced-FaaS-in-China)