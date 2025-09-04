# IPTV-CN: Watch Chinese TV Channels with Jellyfin

**Enhance your Jellyfin media server with a curated list of working Chinese IPTV channels, optimized for smooth streaming!**  For access to the original repository, visit [frankwuzp/iptv-cn](https://github.com/frankwuzp/iptv-cn).

[![GitHub last commit (branch)](https://img.shields.io/github/last-commit/frankwuzp/iptv-cn/main?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![GitHub repo size](https://img.shields.io/github/repo-size/frankwuzp/iptv-cn?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![jsdelivr badge](https://data.jsdelivr.com/v1/package/gh/frankwuzp/iptv-cn/badge)](https://www.jsdelivr.com/package/gh/frankwuzp/iptv-cn)
[![GitHub watchers](https://img.shields.io/github/watchers/frankwuzp/iptv-cn?style=social)](https://github.com/frankwuzp/iptv-cn)

## Key Features:

*   **Working IPTV Channels:** Access a reliable list of Chinese TV channels, tested and optimized for use with Jellyfin.
*   **Province-Specific Options:** Includes channels that are particularly suitable for Guangdong province.
*   **Mobile Signal Source:** Includes mobile signal sources for users.
*   **EPG Guide:** Provides an Electronic Program Guide (EPG) to enhance your viewing experience.
*   **Automatic EPG Updates:** The EPG is automatically updated daily at 1 AM and 6 AM.
*   **Multiple Source Options:** Offers channel lists and EPG files from GitHub and jsDelivr CDN for optimal accessibility.
*   **Easy Integration:** Simple instructions to integrate the channel lists and EPG with your Jellyfin setup.

## Available Files:

*   **`tv-ipv4-cn.m3u`:** General-purpose M3U file for use within China.
*   **`tv-ipv4-cmcc.m3u`:** Mobile signal source, confirmed to be working.
*   **`tv-ipv4-old.m3u`:**  Older channel list (some channels may still work, but can experience delays and buffering).
*   **`guide.xml`:** Electronic Program Guide (EPG) file, automatically updated.
*   **`requirements.txt`:** Dependencies for the Python script used to generate the EPG.

## How to Use with Jellyfin:

1.  **Channel List (M3U) Setup:**

    *   Choose one of the following M3U file sources:
        *   **GitHub:** `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cmcc.m3u`
        *   **jsDelivr CDN (Recommended for mainland China users):** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cmcc.m3u`
    *   In your Jellyfin server, add the chosen URL as your TV channel source.

    ![jellyfin-setting](./image/jellyfin-settings.jpg)

2.  **Electronic Program Guide (EPG) Setup:**

    *   Choose one of the following EPG sources:
        *   **Github:**  `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/guide.xml`
        *   **jsDelivr CDN (Recommended for mainland China users):** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/guide.xml`
        *   **iptv-org:**  `https://iptv-org.github.io/epg/guides/cn/tv.cctv.com.epg.xml`

    *   Enter the selected EPG URL in your Jellyfin server's EPG settings.

    ![jellyfin-epg](./image/jellyfin-epg.jpg)

## References:

*   [BurningC4/Chinese-IPTV](https://github.com/BurningC4/Chinese-IPTV)
*   [SoPudge/kodi_iptv_epg](https://github.com/SoPudge/kodi_iptv_epg)
*   [BurningC4/getepg](https://github.com/BurningC4/getepg)
*   [3mile/cctv_api_get_EPG](https://github.com/3mile/cctv_api_get_EPG)
*   [国内高清直播live - TV001](http://www.tv001.vip/forum.php?mod=viewthread&tid=3)
*   [广东移动某河全套 - 恩山无线论坛](https://www.right.com.cn/forum/thread-6809023-1-1.html)

## Changelog:

*   **2021-11-26:**  Updated notes on non-working sources; added a mobile signal source.
*   **2021-11-23:** Resolved issues with EPG updates and added a new EPG source.
*   **2021-11-22:** Introduced the `guide.xml` EPG file with automated updates (daily at 1 AM and 6 AM).
*   **2021-11-22:**  Channel lists separated into general and Guangdong-specific versions.
*   **2021-11-21:** Initial Release.

**Thank you to the open internet! 🎉🎉🎉**
```
Key improvements:

*   **SEO Optimization:**  Includes keywords like "Chinese IPTV," "Jellyfin," and "EPG" in headings and descriptions.
*   **Clear Structure:** Uses headings, bullet points, and concise language for readability.
*   **Engaging Hook:** The one-sentence hook immediately grabs the reader's attention.
*   **Concise Summarization:**  Provides a summary of the original README, removing unnecessary details.
*   **Emphasis on Benefits:**  Highlights the key features and benefits for users.
*   **Clear Instructions:** Makes the setup process very easy to follow.
*   **Up-to-Date Information:** Clarifies which resources are recommended and which are outdated.
*   **Call to Action:** Encourages users to try the service.
*   **Complete and well-organized** All relevant information is well-ordered
*   **Reorganized for clarity:** The information is presented in a more logical flow.
*   **CDN Recommendation:** Highlights CDN usage for faster access for mainland China users, as well as providing multiple source options.