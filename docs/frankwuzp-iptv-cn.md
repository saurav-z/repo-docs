# IPTV-CN: Free IPTV Resources for China (广东 Supported)

**Enjoy free live TV streaming in China with IPTV-CN, providing reliable IPTV channel lists and EPG guides for your Jellyfin setup.** ([View on GitHub](https://github.com/frankwuzp/iptv-cn))

[![GitHub last commit (branch)](https://img.shields.io/github/last-commit/frankwuzp/iptv-cn/main?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![GitHub repo size](https://img.shields.io/github/repo-size/frankwuzp/iptv-cn?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![jsdelivr badge](https://data.jsdelivr.com/v1/package/gh/frankwuzp/iptv-cn/badge)](https://www.jsdelivr.com/package/gh/frankwuzp/iptv-cn)
[![GitHub watchers](https://img.shields.io/github/watchers/frankwuzp/iptv-cn?style=social)](https://github.com/frankwuzp/iptv-cn)

## Key Features

*   **Free IPTV Channels:** Access a curated list of live TV channels for China, specifically tested for Guangdong province.
*   **Jellyfin Compatible:** Optimized for seamless integration with Jellyfin's live TV feature.
*   **Updated Channel Lists:** Includes multiple `.m3u` channel list files:
    *   `tv-ipv4-cn`: General-purpose channels for China.
    *   `tv-ipv4-cmcc`: Channels sourced from China Mobile (tested and working).
    *   `tv-ipv4-old`:  Archive of older channels.
*   **Electronic Program Guide (EPG):** Includes `guide.xml` for program information, automatically updated daily.
*   **CDN Acceleration:** Offers channel lists and EPG files through jsDelivr CDN for faster access within China.
*   **Easy Integration:** Provides clear instructions and settings examples for configuring within Jellyfin.

## How to Use

### Files Explained

*   `tv-ipv4-cn.m3u`: General IPTV channels for China.
*   `tv-ipv4-cmcc.m3u`: IPTV channels from China Mobile (recommended).
*   `tv-ipv4-old.m3u`:  Older channel list (some channels may be unavailable).
*   `guide.xml`:  Electronic Program Guide (EPG) data, updated daily at 1 AM and 6 AM (UTC+8) via automated GitHub Actions.
*   `requirements.txt`: Python dependencies for `get-epg.py` (EPG generation script).

### Setting up Channel Lists (Example: Guangdong)

1.  **Choose a Channel List:** Select either the `tv-ipv4-cmcc.m3u` file.
2.  **Get the URL:** Obtain the direct URL for your chosen `.m3u` file:
    *   **GitHub:**
        *   `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cmcc.m3u`
    *   **jsDelivr CDN (Recommended for Mainland China):**
        *   `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cmcc.m3u`
3.  **Configure Jellyfin:** In your Jellyfin settings, add the chosen URL to the "TV channels" section.

    ![jellyfin-setting](./image/jellyfin-settings.jpg)

### Setting up the EPG (Example: Guangdong)

1.  **Choose an EPG Source:** Select one of the following `guide.xml` sources.
    *   **GitHub:**
        *   `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/guide.xml`
    *   **jsDelivr CDN (Recommended for Mainland China):**
        *   `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/guide.xml`
    *   **iptv-org**
        *   `https://iptv-org.github.io/epg/guides/cn/tv.cctv.com.epg.xml`
    *   **Mystery Source**
        *   `http://epg.51zmt.top:8000/e.xml`
2.  **Configure Jellyfin:**  Add the URL to the "XMLTV Guide" section within your Jellyfin TV settings.

    ![jellyfin-epg](./image/jellyfin-epg.jpg)

## Resources

*   [BurningC4/Chinese-IPTV](https://github.com/BurningC4/Chinese-IPTV)
*   [SoPudge/kodi_iptv_epg](https://github.com/SoPudge/kodi_iptv_epg)
*   [BurningC4/getepg](https://github.com/BurningC4/getepg)
*   [3mile/cctv_api_get_EPG](https://github.com/3mile/cctv_api_get_EPG)
*   [国内高清直播live - TV001](http://www.tv001.vip/forum.php?mod=viewthread&tid=3)
*   [广东移动某河全套 - 恩山无线论坛](https://www.right.com.cn/forum/thread-6809023-1-1.html)

## Changelog

*   **2021-11-26:** Added China Mobile channels; Updated information on deprecated channels.
*   **2021-11-23:** Fixed EPG update issues; Added new EPG source.
*   **2021-11-22:** Introduced `guide.xml` with automated daily updates (1 AM and 6 AM, UTC+8);  Separated channel lists for general and Guangdong use.
*   **2021-11-21:** Initial release.