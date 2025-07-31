# Speed Up Your Website in China with Enhanced FaaS

**Optimize your Cloudflare, Vercel, and Netlify websites for blazing-fast access in China by simply updating your CNAME record.**  [View the original repo](https://github.com/xingpingcn/enhanced-FaaS-in-China).

[![Stars](https://img.shields.io/github/stars/xingpingcn/enhanced-FaaS-in-China?style=flat)](https://github.com/xingpingcn/enhanced-FaaS-in-China)
[![Last Commit](https://img.shields.io/github/last-commit/xingpingcn/enhanced-FaaS-in-China?display_timestamp=author&style=flat)](https://github.com/xingpingcn/enhanced-FaaS-in-China/commits/main)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fxingpingcn%2Fenhanced-FaaS-in-China&count_bg=%236167ED&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits-since-2024-7-8&edge_flat=false)](https://hits.seeyoufarm.com)

## Key Features

*   **Improved Speed & Stability:** Enhance website access speeds and reliability in China, overcoming potential routing issues.
*   **Easy Setup:** Simply change your CNAME record to a provided optimized record.
*   **Supports Popular Platforms:** Works with Cloudflare, Vercel, and Netlify.
*   **Regularly Updated IP Addresses:** The service automatically monitors and updates IP addresses for optimal performance.

## Usage

To get started, change the CNAME record for your website to one of the following, based on your hosting platform:

> [!NOTE]
>
> The following instructions assume you are modifying the CNAME record for the domain **your users** will visit (e.g., `app.domain.com`).

*   **Vercel:** Use `vercel-cname.xingpingcn.top`
*   **Netlify:** Use `netlify-cname.xingpingcn.top`
*   **Vercel & Netlify (Combined):** Use `verlify-cname.xingpingcn.top`
    > [!IMPORTANT]
    >
    > *Recommended Usage:*  First, set your CNAME record to the official one provided by your hosting platform to ensure SSL/TLS certificate generation.  Then, after your certificate is issued, switch to `verlify-cname.xingpingcn.top`.

*   **Cloudflare:** Use `cf-cname.xingpingcn.top`
    > [!IMPORTANT]
    >
    > *   **Cloudflare Users:** If your domain is also managed by Cloudflare, using this CNAME might result in 403 errors.  Consider transferring your domain management to a non-Cloudflare DNS provider, remove the site from Cloudflare, and then use the provided CNAME record.
    > *   **Cloudflare SaaS and Workers:** If you need to use Cloudflare SaaS features or protect your origin IP, use Cloudflare's SaaS functionalities.

## How to Test & Measure Speed

1.  **After changing your CNAME record, test the speed:**
2.  Use a speed test tool. Remember to include the protocol (e.g., `https://`) in your test. Test from multiple locations as the speed of these test sites can vary.
    ![how2test](img/how2test.png)

## Potential Issues

*   **ISP Restrictions:**  Some ISPs in certain regions (e.g., Quanzhou) may still experience access failures, potentially due to ISP limitations. This can also occur when using the official CNAME records.
*   **Testing Tools:** ITdog.cn may produce inconsistent results. Consider using boce.com, cesu.net, or Alibaba Cloud's speed test tools.

## Why Use This Service?

1.  **Bypass Routing Issues:** Official anycast configurations can route traffic to Southeast Asia, even when a closer US or European route is available.
2.  **Improve Stability:** While official CNAMEs can be fast on average, they may lack stability, leading to inaccessible websites in specific provinces or inconsistent response times.
3.  **Reduce Risk of Outages:** Using a single platform (e.g., Vercel) can result in your site being unavailable across all of China. This service can help mitigate this.

> [!NOTE]
>
> **Optimized Speed Results:**  [See examples of performance improvements in the original README.](https://github.com/xingpingcn/enhanced-FaaS-in-China#why-to-use-it)

## Speed Test Comparisons

> [!IMPORTANT]
> Speed test results may vary. Current results shown may reflect previous data.

<details>
<summary>Click to View Speed Test Results</summary>

<!-- Include speed test images here (e.g., .png files) -->

</details>

## How It Works

This service periodically tests the speed of Cloudflare, Vercel, and Netlify IPs, selecting fast and stable IPs to update DNS A records.  Optimized routes within China are implemented, while foreign traffic uses the default provided A records.

IPs are typically updated every 40 minutes.

### IP Sources

<details>
<summary>Click to View IP Sources</summary>

*   **Vercel:**
    *   [Vercel IP List](https://gist.github.com/ChenYFan/fc2bd4ec1795766f2613b52ba123c0f8)
    *   Official `cname.vercel-dns.com.` A records
*   **Netlify:** Official A records
*   **Cloudflare:**  Various Cloudflare paid user IPs
*   **International IPs:** Default international IPs

```json
{
  "VERCEL": "76.76.21.21",
  "NETLIFY": "75.2.60.5",
  "CF": "japan.com."
}
```

</details>

## Q&A

**Q: Why aren't I seeing the performance improvements shown in the speed test comparisons?**

**A:** Consider the following:

*   **Testing Platform:** Test on multiple speed test platforms to verify consistency.
*   **Domain Issues:** Free subdomains (e.g., .eu.org, .us.kg) or cheap domains (.xyz, .top) may be blocked by some ISPs.
*   **Origin Server:** These tests show the performance of static websites deployed on edge servers.
*   **DNS Resolution:**
    1.  Cloudflare's DNS resolution in mainland China may be slow, adding to the DNS lookup time. Consider using a domestic DNS provider.
    2.  If your domain is managed by Cloudflare or NS1.COM (which doesn't support CNAME records for top-level domains), see [this issue comment](https://github.com/xingpingcn/enhanced-FaaS-in-China/issues/9#issuecomment-2379335329).
    3.  You can manually sync IPs from the JSON files in the repository (updated every 40 minutes) to your A records.
    4.  **Issue with Huawei Cloud DNS:** My Huawei Cloud DNS may incorrectly identify users within China as international users, resulting in the use of the default routes (e.g., japan.com).

**Q: Why isn't my website accessible after changing the CNAME record?**

**A:** Likely causes:

*   **`verlify-cname.xingpingcn.top`:**  This may be due to the CNAME record. Always set the official CNAME first, and then use the combined one after your SSL certificate is generated.
*   **Netlify:** Netlify supports custom certificates. If issues persist, obtain a certificate that renews automatically.
*   **Cloudflare (if domain also managed by Cloudflare):** If you are using `cf-cname.xingpingcn.top` and your domain is managed by Cloudflare, you may experience 403 errors. Consider moving domain management to a non-Cloudflare platform (remove the site from Cloudflare after).
*   **Testing:** If you are experiencing problems on speed testing platforms, refer to the [how to test](#how-to-test--measure-speed) section.

<details>
<summary><b>Q: How is this different from the official CNAME records?</b></summary>
<br>
**A:**

*   Official CNAMEs are fast on average but can be unstable, leading to inaccessibility in some provinces.
*   My CNAME aims to maintain sub-second response times with maximum response times limited to two seconds, and minimizing non-200 status codes in most provinces.

</details>

<details>
<summary><b>Q: Why is the route resolution sometimes inaccurate?</b></summary><br>

**A:** The DNS server's built-in route resolution can sometimes misjudge the user's location. If you want precise route resolution, you can use a different DNS provider such as dnspod, and add the IPs from [Netlify.json](https://raw.githubusercontent.com/xingpingcn/enhanced-FaaS-in-China/main/Netlify.json) or [Vercel.json](https://raw.githubusercontent.com/xingpingcn/enhanced-FaaS-in-China/main/Vercel.json) to A records. Or use `NS1.COM` as your authoritative DNS server, and set up route resolution according to ASN. You can also find out about [China Mainland ASN](https://github.com/xingpingcn/china-mainland-asn).

</details>

<details>
<summary><b>Q: Why do some routes (e.g., Telecom) resolve to the default IP provided by official channels?</b></summary><br>

**A:** It is due to low IP quality in those routes, so this service uses the default IP for that particular route. You can deploy your website in both `vercel` and `netlify` at the same time, and then set up cname resolving to `verlify-cname.xingpingcn.top` to improve fault tolerance. The chance of two platforms failing at the same time in one route is low.

</details>

<details>
<summary><b>Q: Why is the IP list empty in the JSON files for some routes?</b></summary><br>

**A:** See the explanation in the previous answer.

</details>

## Customization

To add a third platform (e.g., Render, Railway), create a new `.py` file in the `platforms_to_test` folder, rewrite the `run_sub()` method, as done in the other files in that folder, and modify the configuration in `config.py`.

## Star History

[![Stargazers over time](https://starchart.cc/xingpingcn/enhanced-FaaS-in-China.svg?background=%23FFFFFF&axis=%23333333&line=%23ff63db)](https://starchart.cc/xingpingcn/enhanced-FaaS-in-China)