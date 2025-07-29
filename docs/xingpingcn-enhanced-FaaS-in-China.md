# Enhanced FaaS in China: Supercharge Your Website Speed and Stability 🚀

**Optimize your Cloudflare, Vercel, and Netlify websites for lightning-fast access in China by simply changing your CNAME record.**  [Visit the original repository for more details](https://github.com/xingpingcn/enhanced-FaaS-in-China).

[![Stars](https://img.shields.io/github/stars/xingpingcn/enhanced-FaaS-in-China?style=flat)](https://github.com/xingpingcn/enhanced-FaaS-in-China)
[![Last Commit](https://img.shields.io/github/last-commit/xingpingcn/enhanced-FaaS-in-China?display_timestamp=author&style=flat)](https://github.com/xingpingcn/enhanced-FaaS-in-China/commits/main)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fxingpingcn%2Fenhanced-FaaS-in-China&count_bg=%236167ED&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits-since-2024-7-8&edge_flat=false)](https://hits.seeyoufarm.com)

## Key Features

*   **Improved Access Speed:** Significantly faster loading times for your websites in China.
*   **Enhanced Stability:** Reduce the likelihood of website access failures.
*   **Simple Setup:** Easy implementation through CNAME record changes.
*   **Platform Support:** Optimized for Cloudflare, Vercel, and Netlify.
*   **Automatic IP Selection:** Continuously monitors and selects the best-performing IPs for optimal performance.

## Usage

This project provides optimized CNAME records to improve access speed and stability for your websites hosted on Cloudflare, Vercel, and Netlify in China. Instead of using the default CNAME records provided by your hosting provider, you can use the following:

> [!NOTE]
>
> Changing the CNAME record means modifying the CNAME record of the domain you wish to access (e.g., app.domain.com) in your authoritative DNS server and replacing it with the optimized CNAME record provided below.

*   **Vercel:** Change your CNAME record to `vercel-cname.xingpingcn.top`
*   **Netlify:** Change your CNAME record to `netlify-cname.xingpingcn.top`
*   **Vercel & Netlify (Combined):** Change your CNAME record to `verlify-cname.xingpingcn.top`

> [!IMPORTANT]
>
> _Recommended DNS Resolution Procedure_:  First, set the CNAME record to the official CNAME provided by your hosting provider.  Once the SSL/TLS certificate is generated, change the CNAME record to the appropriate one above.

*   **Cloudflare:** Change your CNAME record to `cf-cname.xingpingcn.top`

> [!IMPORTANT]
>
> _DNS Resolution Recommendations for Cloudflare_:
>
> 1.  If your domain is hosted on Cloudflare, using this CNAME might result in 403 errors.  Consider hosting your domain on a non-Cloudflare platform, then deleting your site from Cloudflare before using this service.
> 2.  For services like Cloudflare Workers or those that protect your VPS IP with the orange cloud, you must host your domain on Cloudflare and consider using Cloudflare's SaaS feature.  See the documentation on [how to use SaaS Functionality](docs/how2use-SaaS-for-CF/how2use-SaaS-for-CF.md).

## Speed Testing

> [!WARNING]
>
> 1.  Always use a protocol (e.g., `https://`) when testing, and test with multiple speed test websites.
> 2.  Do not test the provided domains, as excessive testing may trigger platform rate limits.

1.  Test after changing the CNAME record.
2.  Use speed test tools.  See the included image for an example.

    ![how2test](img/how2test.png)

## Potential Issues

1.  ~~Access from some ISPs in Zhejiang, Fujian, and Henan may fail~~ Currently, only Quanzhou seems to be blocked (similar to official CNAMEs, possibly due to ISP restrictions).  This is highly dependent on your domain.
2.  ITdog.cn may provide inaccurate results. Consider using boce.com, cesu.net, or Alibaba's speed tests.

## Why Use This?

1.  **Faster Routes:**  Official anycast often routes traffic to Southeast Asia, creating network congestion, while this service utilizes less congested routes to the US or Europe.
2.  **Improved Stability:**  While official CNAMEs can be fast, they may lack stability. This service prioritizes consistent accessibility across provinces.
3.  **Reduced Risk of Outages:** Using optimized CNAMEs helps mitigate the risk of complete website unavailability in China.

> [!NOTE]
>
> **Optimized speed performance**
>
> _Note: Currently, only Quanzhou may be blocked; Speed test results may be outdated; Speed improvements are not significant._
>
> ![vercel-23点晚高峰](img/vercel-2024-9-29-23utc8.png)
> vercel-23 点晚高峰
> ![cf-23点晚高峰](img/cf-2024-9-29-23utc8.png)
> cf-23 点晚高峰

## Speed Test Comparison

> [!IMPORTANT]
> _Note: Currently, only Quanzhou may be blocked; Speed test results may be outdated; Speed improvements are not significant._

<details>
<summary>Click to view results</summary>

![cf-23点晚高峰](img/cf-2024-9-29-23utc8.png)
cf-23 点晚高峰
![cf-22点晚高峰](img/cf-22.5utc8-2024-6-26.png)
cf-22 点晚高峰
![cf-23点晚高峰-官方](img/cf-23utc8-auth.png)
cf-23 点晚高峰-官方
![cf-22点晚高峰-官方](img/cf-22utc8-auth.png)
cf-22点晚高峰-官方
![vercel-23点晚高峰](img/vercel-2024-9-29-23utc8.png)
vercel-23 点晚高峰
![vercel-23点晚高峰-官方](img/vercel-23utc8-auth.png)
vercel-23 点晚高峰-官方
![netlify-23点晚高峰](img/netlify-23utc8.png)
netlify-23 点晚高峰
![netlify-23点晚高峰-官方](img/netlify-23utc8-auth.png)
netlify-23 点晚高峰-官方
![vercel中午](img/vercel-noon.png)
vercel 中午
![vercel中午-官方](img/vercel-noon-auth.png)
vercel 中午-官方
![netlify中午](img/netlify-noon.png)
netlify 中午
![netlify中午-官方](img/netlify-noon-auth.png)
netlify 中午-官方

</details>

## How It Works

This project selects the best-performing IPs from Cloudflare, Vercel, and Netlify, then adds the stable and fast IPs to the A record for your domain.  It optimizes traffic for China's three major network providers and utilizes official A records for international traffic.

The IP addresses are updated approximately every 40 minutes.

### IP Source Details

<details>
<summary>Click to view</summary><br>

*   **Vercel:**
    *   [Vercel IP List](https://gist.github.com/ChenYFan/fc2bd4ec1795766f2613b52ba123c0f8)
    *   Official `cname.vercel-dns.com.` A records
*   **Netlify:**
    *   Official A records
*   **Cloudflare:**
    *   Various Cloudflare paid user optimized IPs.
*   **Default Overseas IPs:**

```json
{
  "VERCEL": "76.76.21.21",
  "NETLIFY": "75.2.60.5",
  "CF": "japan.com."
}
```

</details>

## Q&A

**Q: Why am I not seeing the performance improvements shown in the [Speed Test Comparison](#speed-test-comparison)?**

A:

*   **Testing Platform Issues:**  Compare results across multiple speed test platforms.
*   **Domain Issues:** Free subdomains (e.g., eu.org, us.kg), or inexpensive domains (e.g., .xyz, .top) may be blocked by ISPs. Changing your domain may resolve this.
*   **Origin Server Issues:** The speed tests measure the performance of static web pages hosted on the edge servers.
*   **DNS Resolution Issues:**

    1.  Cloudflare's DNS services may be slower in mainland China.  Using a domestic DNS server may help.
    2.  If your domain is hosted on Cloudflare or NS1.COM (which do not support CNAME records for top-level domains), see this [issue](https://github.com/xingpingcn/enhanced-FaaS-in-China/issues/9#issuecomment-2379335329).
    3.  You can manually update A records with the IPs in the JSON files (updated every 40 minutes).
    4.  **A major problem may be the use of Huawei Cloud DNS, which identifies domestic users as international users and may return default routes (e.g., japan.com).**

**Q: Why is my website inaccessible after setting up your CNAME record?**

A:

*   This is likely related to using `verlify-cname.xingpingcn.top`.  First, set your CNAME record to the official one provided by your hosting provider and wait for the SSL certificate to generate.  Then, try setting it to `verlify-cname.xingpingcn.top` again.  This is because the provided CNAME contains IPs from multiple platforms, which can cause the platform to believe that your domain is not yours.
*   Netlify [supports custom certificates](/netlify_cert/readme.md). If the problem persists, obtain a certificate that auto-renews.

*   If your website is deployed on `cf` and using `cf-cname.xingpingcn.top` and your domain is hosted on Cloudflare, you may encounter 403 errors.  Consider hosting your domain on a non-Cloudflare platform, then deleting your site from Cloudflare before using this service.
*   If you're only experiencing issues in speed tests, see the section on [how to test](#speed-testing).

<details>
<summary><b>Q: What's the difference between your CNAME and the official CNAME?</b></summary>
<br>
A:

*   The official CNAME may be fast on average, but it may also be unstable, with many provinces unable to access the website or with very long response times.
*   This CNAME may not always be the fastest, but it attempts to keep average response times under 1 second, with the longest response times ideally under 2 seconds and with no more than two provinces returning non-200 status codes.
</details>
<details>
<summary><b>Q: Why is route-based resolution inaccurate?</b></summary><br>

A: This project uses the built-in route resolution provided by the authoritative DNS server, which may lead to incorrect results.  For more precise route-based resolution, consider using a different DNS server (e.g., dnspod) and adding the IPs from [Netlify.json](https://raw.githubusercontent.com/xingpingcn/enhanced-FaaS-in-China/main/Netlify.json) or [Vercel.json](https://raw.githubusercontent.com/xingpingcn/enhanced-FaaS-in-China/main/Vercel.json) to your A records. Alternatively, use `NS1.COM` as your authoritative DNS server and configure route-based resolution based on `ASN`. You can also refer to my [ASN list](https://github.com/xingpingcn/china-mainland-asn).

</details>

<details>
<summary><b>Q: Why does the DNS A record for some routes (e.g., Telecom) resolve to the official default IP?</b></summary><br>

A: This is due to poor IP quality on those routes. The service temporarily stops resolving those routes and uses the official default IP. You can increase fault tolerance by deploying the website on both `vercel` and `netlify` and setting the CNAME to `verlify-cname.xingpingcn.top`.  The probability of failure for both platforms on the same line at the same time is very low.

</details>
<details>
<summary><b>Q: Why are some routes in the JSON file an empty list?</b></summary><br>

A: See the previous answer.

</details>

## Customization

If you want to add a third platform (e.g., `render`, `railway`), you'll need to prepare your own speed testing tool and a domain, rewrite `crawler.py`, create a new `.py` file in the `platforms_to_test` directory, and rewrite the `run_sub()` method as other files. Finally, modify the relevant configurations in the `config.py` file.

## Star History

[![Stargazers over time](https://starchart.cc/xingpingcn/enhanced-FaaS-in-China.svg?background=%23FFFFFF&axis=%23333333&line=%23ff63db)](https://starchart.cc/xingpingcn/enhanced-FaaS-in-China)