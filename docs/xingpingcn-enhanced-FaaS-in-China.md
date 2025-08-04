# Enhance Website Access Speed in China with Optimized CNAME Records

**Instantly improve the access speed and stability of your Cloudflare, Vercel, or Netlify hosted websites in China by using custom CNAME records.**  [View the original repository](https://github.com/xingpingcn/enhanced-FaaS-in-China)

[![Stars](https://img.shields.io/github/stars/xingpingcn/enhanced-FaaS-in-China?style=flat)](https://github.com/xingpingcn/enhanced-FaaS-in-China)
[![Last Commit](https://img.shields.io/github/last-commit/xingpingcn/enhanced-FaaS-in-China?display_timestamp=author&style=flat)](https://github.com/xingpingcn/enhanced-FaaS-in-China)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fxingpingcn%2Fenhanced-FaaS-in-China&count_bg=%236167ED&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits-since-2024-7-8&edge_flat=false)](https://hits.seeyoufarm.com)

## Key Features

*   **Improved Access Speed:** Optimized CNAME records to route traffic through faster and more stable routes within China.
*   **Enhanced Stability:** Addresses the instability issues often encountered with official anycast solutions in China.
*   **Simple Implementation:** Easy to use - just change your CNAME record.
*   **Platform Support:** Works with Cloudflare, Vercel, and Netlify.

## Usage

To improve the access speed of your website, simply update the CNAME record for your domain (e.g., `app.yourdomain.com`) in your DNS settings to the appropriate CNAME record provided below.  **It is recommended to first set your CNAME to the official platform's value until an SSL/TLS certificate is generated.**

*   **Vercel:**  Use `vercel-cname.xingpingcn.top`
*   **Netlify:** Use `netlify-cname.xingpingcn.top`
*   **Vercel & Netlify (Combined):** Use `verlify-cname.xingpingcn.top`
*   **Cloudflare:**  Use `cf-cname.xingpingcn.top`

    *   **Important for Cloudflare Users:** If your domain uses Cloudflare's DNS, using `cf-cname.xingpingcn.top` *may* result in 403 errors.  Consider hosting your DNS with a non-Cloudflare provider for optimal results. Alternatively, utilize Cloudflare's SaaS feature, as detailed in the documentation: [How to use SaaS function](docs/how2use-SaaS-for-CF/how2use-SaaS-for-CF.md)

## Testing Website Speed

1.  **After Changing CNAME:** Test your website's loading speed after you update the CNAME records.
2.  **Online Speed Testing Tools:** Use online speed test tools, specifying the protocol (HTTP/HTTPS).

    ![How to Test](img/how2test.png)

## Potential Issues and Troubleshooting

*   **ISP-Specific Problems:** Occasional access failures may occur with some ISPs (e.g. certain networks in Quanzhou).
*   **Testing Tools:** Results from testing tools may vary. Consider using multiple testing sites.
*   **SSL/TLS Certificate:** To generate the SSL certificate, start by using the original CNAME record for your platform, then change to the custom one provided.
*   **Cloudflare Users:** If your domain is hosted with Cloudflare and you're using the `cf-cname.xingpingcn.top`, ensure you understand the potential for 403 errors.
*   **Domain Name Issues:** Problems can occur with free or cheap domain names.

## Why Use This?

*   **Bypass Routing Issues:** Official anycast solutions may route traffic to Southeast Asia, causing slower access.
*   **Improve Stability:** Address inconsistencies in official CNAME performance, especially in China.
*   **Reduce Single Point of Failure:** Minimize the impact of potential website unavailability within China.

## Speed Test Results

_Note:  Speed test results are from previous tests and may not reflect real-time performance._

*   [View speed comparison results](#测速对比)

## How It Works

This project works by:

1.  **IP Selection:** Regularly testing IP addresses for Cloudflare, Vercel, and Netlify.
2.  **Route Optimization:** Selecting the most stable and fastest IPs and adding them to the domain's A records.
3.  **Real-time Updates:** The A records are updated approximately every 40 minutes.

## Q&A

*   **Q: Why am I not seeing the speed improvements in the speed test results?**
    *   Consider factors like the testing platform, your domain, and DNS caching.
*   **Q: Why is my website not accessible after changing the CNAME?**
    *   Follow the SSL certificate generation procedure, as explained above.
*   **Q: What's the difference between the provided CNAME records and the official ones?**
    *   Optimized for speed and stability within China, often offering better performance than the official anycast solutions.
*   **Q: Why are my DNS A records pointing to the default IP addresses for some routes?**
    *   This can occur when other IPs are of lower quality.

*   [Further Q&A is in the original repository, under the Q&A section.](#Q&A)

## Customization

If you want to add support for other platforms (e.g., Render, Railway), you'll need to:

1.  Prepare a speed testing tool and a domain.
2.  Modify `crawler.py`.
3.  Create a `.py` file in `platforms_to_test`, following existing examples.
4.  Modify `config.py`.

## Star History

[![Stargazers over time](https://starchart.cc/xingpingcn/enhanced-FaaS-in-China.svg?background=%23FFFFFF&axis=%23333333&line=%23ff63db)](https://starchart.cc/xingpingcn/enhanced-FaaS-in-China)