# Enhanced FaaS Access Speed & Stability in China

**Supercharge your Cloudflare, Vercel, and Netlify websites in China with faster access speeds and improved stability by simply changing your CNAME records.** ([Original Repository](https://github.com/xingpingcn/enhanced-FaaS-in-China))

![Stars](https://img.shields.io/github/stars/xingpingcn/enhanced-FaaS-in-China?style=flat)
![Last Commit](https://img.shields.io/github/last-commit/xingpingcn/enhanced-FaaS-in-China?display_timestamp=author&style=flat)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fxingpingcn%2Fenhanced-FaaS-in-China&count_bg=%236167ED&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits-since-2024-7-8&edge_flat=false)](https://hits.seeyoufarm.com)

## Key Features:

*   **Improved Access Speed:** Experience faster website loading times for users in China.
*   **Enhanced Stability:** Benefit from more reliable access, reducing the likelihood of outages.
*   **Simplified Setup:** Easy implementation by simply changing your DNS CNAME records.
*   **Optimized for Cloudflare, Vercel, and Netlify:** Specifically tailored for popular FaaS platforms.
*   **Dynamic IP Selection:**  The service automatically selects and uses the best-performing IPs.

## Getting Started

This guide provides instructions on how to change your CNAME records to enhance website performance in China.

**How to Use:**  Update the CNAME record of your domain (e.g., `app.domain.com`) with the appropriate value based on your hosting platform:

*   **Vercel:** `vercel-cname.xingpingcn.top`
*   **Netlify:** `netlify-cname.xingpingcn.top`
*   **Vercel & Netlify (Combined):** `verlify-cname.xingpingcn.top`
*   **Cloudflare:** `cf-cname.xingpingcn.top`

    *   **Important for Cloudflare users:** If your domain is hosted on Cloudflare, using `cf-cname.xingpingcn.top` might result in 403 errors.  Consider transferring your domain to a different DNS provider and then removing the site from Cloudflare before using this service, or using Cloudflare's SaaS functionality.

> [!NOTE]
>
> Before changing the CNAME records, it's recommended that you first create an SSL/TLS certificate from the host.

## Troubleshooting & FAQs

**Q: Why is the speed not as good as the benchmark results?**

*   **Testing Platform Issues:** Test with multiple speed test sites to ensure consistent results.
*   **Domain Issues:** Free or cheap domains (e.g., `.xyz`, `.top`) might be blocked. Consider using a different domain.
*   **Origin Server Issues:** The test results are for static web pages deployed on edge servers.
*   **DNS Resolution Issues:** If using Cloudflare's DNS, the DNS resolution speed might be slow in China. Also, if you're using NS1.COM, you might not be able to add CNAME records to top-level domains. See [this issue](https://github.com/xingpingcn/enhanced-FaaS-in-China/issues/9#issuecomment-2379335329) for more information.

**Q: Why is my website not accessible after changing the CNAME records?**

*   If you are using `verlify-cname.xingpingcn.top`, make sure that you've created an SSL/TLS certificate first.
*   If you are using `cf-cname.xingpingcn.top` and your domain is hosted on Cloudflare, you may encounter 403 errors. Consider using a different DNS provider.

## Performance Comparisons

> [!IMPORTANT]
> The speed test results may not be up to date

<details>
<summary><b>Click to view speed test results</b></summary>

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

This service periodically tests the speeds of IPs for Cloudflare, Vercel, and Netlify, and automatically updates the A records with the fastest and most stable IPs. Updates are performed approximately every 40 minutes.

### IP Sources

<details>
<summary><b>Click to view</b></summary><br>

*   **Vercel:**
    *   [vercel ip](https://gist.github.com/ChenYFan/fc2bd4ec1795766f2613b52ba123c0f8)
    *   Official `cname.vercel-dns.com.` A records
*   **Netlify:**
    *   Official A records
*   **Cloudflare:**
    *   Optimized IPs for Cloudflare paying users
*   **Default Overseas IPs:**
    ```json
    {
      "VERCEL": "76.76.21.21",
      "NETLIFY": "75.2.60.5",
      "CF": "japan.com."
    }
    ```

</details>

##  Further Information

*   **Q: How does this service differ from the official CNAME records?**

    *   The official CNAMEs may offer fast average speeds but can be unstable, with accessibility issues in certain provinces. This service prioritizes consistent performance and aims to keep response times under 1 second (with a maximum of 2 seconds), with minimal failures.

*   **Q: Why isn't route-based DNS resolution accurate?**

    *   The route-based DNS resolution might be inaccurate. For precise route-based resolution, consider using a DNS provider like dnspod.

*   **Q: Why are some routes (e.g., Telecom) resolving to the default IPs?**

    *   Routes with poor IP quality are temporarily using the default IPs.

*   **Q: Why are some routes empty in the JSON files?**
    *   As above.

## Customization

If you want to add support for additional platforms like `render` or `railway`, you will need to:

1.  Create a speed testing setup.
2.  Write a `.py` file in the `platforms_to_test` directory, mimicking the structure of other files.
3.  Modify the `config.py` file.

## Star History

[![Stargazers over time](https://starchart.cc/xingpingcn/enhanced-FaaS-in-China.svg?background=%23FFFFFF&axis=%23333333&line=%23ff63db)](https://starchart.cc/xingpingcn/enhanced-FaaS-in-China)