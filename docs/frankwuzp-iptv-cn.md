# IPTV-CN: Free IPTV Resources for China (Updated & Optimized)

**Enjoy free and reliable IPTV streams for Chinese TV channels, optimized for Jellyfin, with updated links and EPG.** ([View the original repo](https://github.com/frankwuzp/iptv-cn))

[![GitHub last commit (branch)](https://img.shields.io/github/last-commit/frankwuzp/iptv-cn/main?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![GitHub repo size](https://img.shields.io/github/repo-size/frankwuzp/iptv-cn?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![jsdelivr](https://data.jsdelivr.com/v1/package/gh/frankwuzp/iptv-cn/badge)](https://www.jsdelivr.com/package/gh/frankwuzp/iptv-cn)
[![GitHub watchers](https://img.shields.io/github/watchers/frankwuzp/iptv-cn?style=social)](https://github.com/frankwuzp/iptv-cn)

## Key Features:

*   **Free IPTV Streams:** Access a curated list of free IPTV channels, including Chinese TV channels.
*   **Optimized for Jellyfin:**  Designed for seamless integration with Jellyfin's live TV feature.
*   **Updated Resources:**  Provides the latest working IPTV links and EPG (Electronic Program Guide) files.
*   **Multiple Sources:** Offers channel lists and EPG from Github and CDN (Content Delivery Network) for faster and more reliable streaming.
*   **Automatic EPG Updates:** The EPG is automatically updated daily to ensure accurate program information.
*   **CDN Support:**  Leverages jsDelivr CDN for faster access, particularly for users in mainland China.
*   **Mobile Support:** Includes a mobile IPTV source for easier streaming experience.

## Getting Started

### Available Files

*   `tv-ipv4-cn.m3u`:  General purpose IPTV playlist for Mainland China.
*   `tv-ipv4-cmcc.m3u`:  IPTV playlist optimized for China Mobile users (tested and working).
*   `tv-ipv4-old.m3u`: Older IPTV list from [BurningC4/Chinese-IPTV](https://github.com/BurningC4/Chinese-IPTV) - may have some channels working.
*   `guide.xml`: EPG (Electronic Program Guide) file, automatically updated daily via GitHub Actions at 1 AM and 6 AM UTC.
*   `requirements.txt`: Dependencies for the `get-epg.py` Python script (used for EPG generation).

### How to Use with Jellyfin

1.  **Choose Your Channel List:**
    *   **Recommended (China Mobile):**  `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cmcc.m3u` or `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cmcc.m3u` (CDN - recommended for mainland China).
    *   **General (Mainland China):** `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cn.m3u` or `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cn.m3u` (CDN - recommended for mainland China).
    *   Save the .m3u file to your device or copy one of the above URLs.

2.  **Configure Jellyfin:**
    *   In Jellyfin, go to Live TV settings.
    *   Add a new TV provider and select "M3U Tuner."
    *   Paste the URL of your chosen channel list into the "M3U URL" field.

    ![jellyfin-setting](./image/jellyfin-settings.jpg)

3.  **Choose Your EPG Source (Four Options):**
    *   `http://epg.51zmt.top:8000/e.xml` (Recommended)
    *   `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/guide.xml`
    *   `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/guide.xml` (CDN - recommended for mainland China)
    *   `https://iptv-org.github.io/epg/guides/cn/tv.cctv.com.epg.xml`

    *   Enter the URL of your chosen EPG source in the Jellyfin Live TV settings, under "XMLTV Guide URL"
    ![jellyfin-epg](./image/jellyfin-epg.jpg)

##  References

*   [BurningC4/Chinese-IPTV](https://github.com/BurningC4/Chinese-IPTV)
*   [SoPudge/kodi_iptv_epg](https://github.com/SoPudge/kodi_iptv_epg)
*   [BurningC4/getepg](https://github.com/BurningC4/getepg)
*   [3mile/cctv_api_get_EPG](https://github.com/3mile/cctv_api_get_EPG)
*   [国内高清直播live - TV001](http://www.tv001.vip/forum.php?mod=viewthread&tid=3)
*   [广东移动某河全套 - 恩山无线论坛](https://www.right.com.cn/forum/thread-6809023-1-1.html)

**Thank you to the open-source community! 🎉🎉🎉**

## Changelog

*   211126: Marked non-working streams, added China Mobile source.
*   211123: Fixed EPG update issues, added new EPG source.
*   211122: Added EPG guide file (`guide.xml`) with automatic updates (daily at 1 AM and 6 AM UTC).
*   211122: Divided into general and Guangdong province-specific versions.
*   211121: Initial release.
```

Key improvements and SEO considerations:

*   **Clear, Concise, and SEO-Friendly Title:**  Uses keywords like "IPTV," "China," "Free," "Jellyfin," and "Updated" to attract relevant search traffic.
*   **One-Sentence Hook:** Grabs attention immediately.
*   **Bulleted Key Features:** Makes the information easy to scan and highlights the value proposition.
*   **Actionable "Getting Started" Section:** Guides users through the setup process.
*   **Simplified Instructions:** Streamlined the "How to Use" section for clarity.
*   **CDN Emphasis:**  Highlights the benefit of the CDN for users in mainland China.
*   **Updated Information:**  Removed outdated information and included current working sources.
*   **Call to Action:** Includes the original repo link.
*   **Clean Formatting:** Uses Markdown for better readability.
*   **Keyword Optimization:**  Uses relevant keywords throughout the document.
*   **Complete and Updated:**  Incorporated all the information from the original README, while improving clarity.