# IPTV-CN: Free IPTV Channels for Jellyfin (China)

**Access free, reliable Chinese IPTV channels directly within your Jellyfin media server using this regularly updated resource.** ([View the Original Repo](https://github.com/frankwuzp/iptv-cn))

[![Last Commit](https://img.shields.io/github/last-commit/frankwuzp/iptv-cn/main?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![Repo Size](https://img.shields.io/github/repo-size/frankwuzp/iptv-cn?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![jsDelivr](https://data.jsdelivr.com/v1/package/gh/frankwuzp/iptv-cn/badge)](https://www.jsdelivr.com/package/gh/frankwuzp/iptv-cn)
[![GitHub Watchers](https://img.shields.io/github/watchers/frankwuzp/iptv-cn?style=social)](https://github.com/frankwuzp/iptv-cn)

## Key Features

*   **Up-to-Date Channel Lists:** Provides `.m3u` files with working IPTV channels, regularly tested and updated.
*   **Optimized for Jellyfin:** Designed for seamless integration with Jellyfin's live TV feature.
*   **Multiple Channel Sources:** Offers both general and mobile channel lists for wider compatibility.
*   **EPG (Electronic Program Guide):** Includes an automatically updated `guide.xml` file for program information.
*   **CDN Support:** Uses jsDelivr CDN for faster access, especially for users in mainland China.

## Getting Started

### Files Overview

*   `tv-ipv4-cn.m3u`:  General IPTV channel list for China.
*   `tv-ipv4-cmcc.m3u`:  Mobile (CMCC) channel list (tested and working).
*   `tv-ipv4-old.m3u`: Older channel list, may have some channels still working.
*   `guide.xml`:  Electronic Program Guide (EPG) file, automatically updated daily.
*   `requirements.txt`: Python dependencies for the EPG generation script (`get-epg.py`).

### How to Use with Jellyfin

1.  **Channel List (M3U):**

    *   **Choose a Source:**  You can use the general `tv-ipv4-cn.m3u` or the `tv-ipv4-cmcc.m3u` for mobile.
    *   **GitHub:**  `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cmcc.m3u` (or `tv-ipv4-cn.m3u`)
    *   **jsDelivr CDN (Recommended for Mainland China):**  `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cmcc.m3u` (or `tv-ipv4-cn.m3u`)
    *   **Method:** Copy the URL you choose.
    *   **Jellyfin Configuration:**  Go to your Jellyfin server settings, and within the "Live TV" section, add the M3U URL to the TV provider.

2.  **EPG Guide (XML):**

    *   **Choose a Source:**
    *   **Mystery Source:** `http://epg.51zmt.top:8000/e.xml`
    *   **GitHub:** `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/guide.xml`
    *   **jsDelivr CDN (Recommended for Mainland China):**  `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/guide.xml`
    *   **iptv-org:** `https://iptv-org.github.io/epg/guides/cn/tv.cctv.com.epg.xml`
    *   **Method:** Copy the URL you choose.
    *   **Jellyfin Configuration:** Add the XML URL to the EPG provider within the "Live TV" section of your Jellyfin server settings.

    ![jellyfin-settings](./image/jellyfin-settings.jpg)

    ![jellyfin-epg](./image/jellyfin-epg.jpg)

## References

*   [BurningC4/Chinese-IPTV](https://github.com/BurningC4/Chinese-IPTV)
*   [SoPudge/kodi_iptv_epg](https://github.com/SoPudge/kodi_iptv_epg)
*   [BurningC4/getepg](https://github.com/BurningC4/getepg)
*   [3mile/cctv_api_get_EPG](https://github.com/3mile/cctv_api_get_EPG)
*   [国内高清直播live - TV001](http://www.tv001.vip/forum.php?mod=viewthread&tid=3)
*   [广东移动某河全套 - 恩山无线论坛](https://www.right.com.cn/forum/thread-6809023-1-1.html)

## Changelog

*   **211126:** Added CMCC channel source; noted unworking sources.
*   **211123:** Fixed EPG update issues & added a new EPG source.
*   **211122:** Added automatic EPG updates (daily at 1 AM and 6 AM).
*   **211122:** Separated channel lists into general and Guangdong-specific versions.
*   **211121:** Initial release.

**Enjoy watching free Chinese IPTV channels! 🎉**
```
Key improvements and optimizations:

*   **Clear and Concise Headings:** Uses clear and descriptive headings like "Key Features" and "Getting Started."
*   **Bulleted Key Features:**  Highlights the essential aspects of the project using bullet points, making it easy to scan and understand.
*   **SEO Optimization:** Includes relevant keywords like "IPTV," "Jellyfin," "China," and "free" throughout the document.
*   **Concise Language:** Streamlines the language for better readability.
*   **Call to Action:**  Encourages use and provides step-by-step instructions.
*   **CDN Emphasis:**  Highlights the jsDelivr CDN as the preferred option for mainland Chinese users, which is crucial for performance.
*   **Updated Information:**  Reflects the changes in the original README regarding channel availability.
*   **Clear Instructions:** Provides easy-to-follow instructions on setting up both the M3U and EPG files within Jellyfin.
*   **Changelog Included:** Retained the changelog for user awareness.
*   **One-Sentence Hook:**  Added a strong introductory sentence to capture interest and provide context.
*   **Simplified Formatting:** Improved the readability of the text by utilizing a consistent style.