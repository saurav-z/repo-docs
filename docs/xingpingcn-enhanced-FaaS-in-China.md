# Enhance Website Speed & Stability in China with Optimized CNAME Records

**Instantly improve the access speed and stability of your Cloudflare, Vercel, or Netlify hosted websites in China by simply changing your CNAME records.**  [View the original repository](https://github.com/xingpingcn/enhanced-FaaS-in-China)

[![Stars](https://img.shields.io/github/stars/xingpingcn/enhanced-FaaS-in-China?style=flat)](https://github.com/xingpingcn/enhanced-FaaS-in-China)
[![Last Commit](https://img.shields.io/github/last-commit/xingpingcn/enhanced-FaaS-in-China?display_timestamp=author&style=flat)](https://github.com/xingpingcn/enhanced-FaaS-in-China/commits/main)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fxingpingcn%2Fenhanced-FaaS-in-China&count_bg=%236167ED&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits-since-2024-7-8&edge_flat=false)](https://hits.seeyoufarm.com)

## Key Features:

*   **Improved Speed and Stability:** Optimized CNAME records to route traffic through faster and more reliable routes within China.
*   **Easy Implementation:** Simple CNAME record changes for quick integration with your existing hosting setup.
*   **Platform Support:** Works with Cloudflare, Vercel, and Netlify.
*   **Automated IP Selection:** Dynamically selects the best-performing IP addresses for optimal performance.

## Usage

This guide explains how to configure your DNS settings to enhance access to your websites within China.  The core concept involves changing your CNAME records to point to optimized domains.

> [!NOTE]
>
> When I say change the cname record to xxx, it means changing the CNAME record of the domain that the **user** wants to access - for example, app.domain.com - to the preferred CNAME record of the corresponding platform in your authoritative DNS server.
>
> _Example 1:_ I need to speed up access to the `blog.domain.com` blog deployed on vercel. First, change the cname record of `blog.domain.com` to the domain name provided by the official, and then go back to the vercel console to check whether the ssl certificate is generated. After it is generated, change the cname record to `vercel-cname.xingpingcn.top`
>
> _Example 2:_ I need to speed up the website built on my vps, I need to use cf as CDN and protect the source IP. [How to use saas function](docs/how2use-SaaS-for-CF/how2use-SaaS-for-CF.md)

**Follow these instructions based on your hosting provider:**

*   **Vercel:**
    *   Change your CNAME record to: `vercel-cname.xingpingcn.top`
*   **Netlify:**
    *   Change your CNAME record to: `netlify-cname.xingpingcn.top`
*   **Netlify and Vercel:**
    *   Change your CNAME record to: `verlify-cname.xingpingcn.top`

> [!IMPORTANT]
>
> _Suggestions for using this DNS resolution_: First, change the cname record to the cname provided by the official, and then change the cname record to `verlify-cname.xingpingcn.top` after the `ssl/tls certificate` is generated.

*   **Cloudflare:**
    *   Change your CNAME record to: `cf-cname.xingpingcn.top`

> [!IMPORTANT]
>
> _Suggestions for using this DNS resolution_:
>
> 1.  If your domain is hosted on cloudflare, then using this cname is very likely to encounter 403. It is recommended that you host your domain on a non-cloudflare platform, and then delete your site in the cf platform, and then use it.
> 2.  If some services, such as cf worker, open the orange cloud protection vps's IP, you must host your domain on cf, then it is recommended that you use cf's saas function. [How to use saas function](docs/how2use-SaaS-for-CF/how2use-SaaS-for-CF.md)

### How to Test Speed

> [!WARNING]
>
> 1.  No matter which method you use to test, you must add the protocol, and then test on multiple speed test websites, because the speed test website itself will also go crazy from time to time
> 2.  Don't test these domain names of mine, because there will be false negatives, and too many people will be considered ddos by the corresponding platform, although it has no effect on dns resolution.

1.  You can test after changing the cname record
2.  You can also fill in the relevant information as shown in the figure below, and then test the speed
    ![how2test](img/how2test.png)

### Potential Issues

1.  ~~Individual ISPs in Zhejiang, Fujian, and Henan may experience access failures~~ Currently, it seems that only Quanzhou is blocked (the official cname also has the same problem, which may be caused by ISP restrictions. It has a great relationship with your domain name).
2.  For the selection of speed test tools, the results measured by itdog.cn are a bit problematic (there will be large areas of red, the reason is unknown), you can try to use boce.com, cesu.net, Alibaba Cloud dial-up test, etc.

## Why Use This?

1.  **Optimized Routing:** Official Anycast often routes traffic to Southeast Asia, resulting in high latency. This solution leverages potentially faster routes in the US or Europe.
2.  **Enhanced Stability:** While official CNAMEs might be fast on average, they can lack stability, leading to accessibility issues in various provinces. This solution provides more consistent performance.
3.  **Reduced Risk of Outages:**  Mitigates the risk of complete website unavailability in China, which can happen if you rely on a single platform (e.g., Vercel) and it becomes blocked.

> [!NOTE]
>
> **This is the optimized speed**
>
> _Note: Currently, only Quanzhou seems to be blocked (red); the speed measurement results have not been updated in time, and the current display is the previous speed measurement results; the speed measurement speed has not changed much_
>
> ![vercel-23点晚高峰](img/vercel-2024-9-29-23utc8.png)
> vercel-23 点晚高峰
> ![cf-23点晚高峰](img/cf-2024-9-29-23utc8.png)
> cf-23 点晚高峰

## Speed Test Comparison

> [!IMPORTANT]
> _Note: Currently, only Quanzhou seems to be blocked (red); the speed measurement results have not been updated in time, and the current display is the previous speed measurement results; the speed measurement speed has not changed much_

<details>
<summary>Click to View Results</summary>

![cf-23点晚高峰](img/cf-2024-9-29-23utc8.png)
cf-23 点晚高峰
![cf-22点晚高峰](img/cf-22.5utc8-2024-6-26.png)
cf-22 点晚高峰
![cf-23点晚高峰-官方](img/cf-23utc8-auth.png)
cf-23 点晚高峰-官方
![cf-22点晚高峰-官方](img/cf-22utc8-auth.png)
cf-22 点晚高峰-官方
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

This solution selects the fastest and most stable IP addresses for Cloudflare, Vercel, and Netlify by continuously testing their speeds.  These optimized IPs are then used in the A records for your domain. Optimized IPs are updated approximately every 40 minutes.

### IP Sources

<details>
<summary>Click to View</summary><br>

*   **Vercel:**
    *   [vercel ip](https://gist.github.com/ChenYFan/fc2bd4ec1795766f2613b52ba123c0f8)
    *   Official `cname.vercel-dns.com.` A records
*   **Netlify:**
    *   A records from the official link
*   **Cloudflare:**
    *   Optimized IPs from various Cloudflare paid users

*   **Default IPs for outside of China:**

```json
{
  "VERCEL": "76.76.21.21",
  "NETLIFY": "75.2.60.5",
  "CF": "japan.com."
}
```

</details>

## Q&A

**Q: Why isn't the access speed as fast as the [speed test comparison](#speed-test-comparison) results after setting your CNAME resolution?**<br>
A:

*   Speed test platform issues: Compare the results of multiple speed test platforms to see if the results of each speed test platform are consistent
*   Domain name issues: If the domain name you are testing is a free second-level domain name (eu.org, us.kg), or.xyz,.top and other cheap domain names, it may be blocked due to the whitelist mechanism of the operator, in this case, you can only solve it by changing the domain name (or believe in metaphysics and wait for your domain name to be moved into the whitelist).
*   Return source issue: The test comparison shows the speed test effect of static web pages deployed on edge servers. The test uses cf page and static web pages deployed on vercel and netlify.
*   DNS resolution issues:

1.  If you use cf's dns service, the dns resolution speed in the mainland is slow, and you need to add another dns recursive resolution due to cname, which will be even slower. If possible, please host your domain name on a domestic authoritative dns server.
2.  If your domain name is hosted on cf or NS1.COM, which does not support adding cname records to the top-level domain name, please see [here#9](https://github.com/xingpingcn/enhanced-FaaS-in-China/issues/9#issuecomment-2379335329)
3.  Or you can open the three json files in the root directory of this repo, which contain the real-time updated ip. If you want, you can try to synchronize it to your a record. The ips in the repo are generally updated every 40 minutes.
4.  **The biggest problem that may appear in the end is the Huawei Cloud DNS I used. It was originally a domestic user to visit the website, but it was recognized as a foreign user by the Huawei Cloud DNS, and then the default route-such as japan.com-was resolved.**

**Q: Why can't the website be accessed after setting your CNAME resolution?**<br>
A:

*   This is most likely caused by using `verlify-cname.xingpingcn.top`. You need to change the CNAME record to the link provided by the official first, and then reset it after the SSL certificate is generated. This is because this resolution contains the IP of the two platforms, and the platform will obtain one of the two IPs each time it is accessed, so it is considered that the domain name you filled in on the platform does not belong to you. However, once the certificate is generated, the certificate will be cached on the platform.
*   netlify[supports uploading your own certificate](/netlify_cert/readme.md). If it still doesn't work, apply for a certificate that can be automatically renewed.

*   If your website is deployed on `cf` and you use `cf-cname.xingpingcn.top`, if your domain is hosted on cloudflare, then using this cname is very likely to encounter 403. It is recommended that you host your domain on a non-cloudflare platform, such as Huawei Cloud, **and then delete your site in the cf platform**, and then use it.
*   If you are only having problems on the speed test platform, you may need to see [how to test speed](#how-to-test-speed)

<details>
<summary><b>Q: What's the difference with the official cname?</b></summary>
<br>
A:

*   The official cname is sometimes very fast on average, but it lacks stability, and there will be a situation that several provinces cannot access it, or the response time of individual provinces is very long
*   And my cname may not be the fastest on average, but the average response speed tries to stay within 1 second, the longest response time is controlled within 2 seconds, and the provinces that return non-200 status codes are less than or equal to 2

</details>
<details>
<summary><b>Q: Why is the route resolution inaccurate?</b></summary><br>

A: I use the route resolution that comes with the authoritative DNS server, which may be misjudged. If you want more accurate route resolution, you can choose other DNS servers-such as dnspod-and add the IP in [Netlify.json](https://raw.githubusercontent.com/xingpingcn/enhanced-FaaS-in-China/main/Netlify.json) or [Vercel.json](https://raw.githubusercontent.com/xingpingcn/enhanced-FaaS-in-China/main/Vercel.json) to the A record. Or use `NS1.COM` as the authoritative DNS server and set route resolution according to `ASN`. You can see my [ASN list](https://github.com/xingpingcn/china-mainland-asn).

</details>

<details>
<summary><b>Q: Why is the DNS A record resolution of some routes (such as Telecom) the default IP provided by the official?</b></summary><br>

A: This is because the quality of other IPs on this route is poor, so the resolution of this route is temporarily stopped, and the default IP provided by the official is used instead. You can deploy the website on `vercel` and `netlify` at the same time, and change the cname resolution to `verlify-cname.xingpingcn.top` to improve fault tolerance. The probability of the same line of the two platforms failing at the same time is much lower.

</details>
<details>
<summary><b>Q: Why is there an empty list of routes in the json file?</b></summary><br>

A: Same as above

</details>

## Customization

If you want to customize, such as adding a third platform, such as `render`, `railway`, etc., you need to prepare the speed test tool and a domain name, rewrite `crawler.py`, create a `.py` file in `platforms_to_test`, rewrite the `run_sub()` method by imitating other files in the folder, and finally modify the relevant configuration of the `config.py` file.

## Star History

[![Stargazers over time](https://starchart.cc/xingpingcn/enhanced-FaaS-in-China.svg?background=%23FFFFFF&axis=%23333333&line=%23ff63db)](https://starchart.cc/xingpingcn/enhanced-FaaS-in-China)