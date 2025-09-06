# IPTV-CN: Free Live TV Channels for Jellyfin (China)

Easily stream free, live TV channels, specifically optimized for users in China, using Jellyfin and this repository. [(View Original Repo)](https://github.com/frankwuzp/iptv-cn)

[![GitHub last commit](https://img.shields.io/github/last-commit/frankwuzp/iptv-cn/main?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![GitHub repo size](https://img.shields.io/github/repo-size/frankwuzp/iptv-cn?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![jsdelivr](https://data.jsdelivr.com/v1/package/gh/frankwuzp/iptv-cn/badge)](https://www.jsdelivr.com/package/gh/frankwuzp/iptv-cn)
[![GitHub watchers](https://img.shields.io/github/watchers/frankwuzp/iptv-cn?style=social)](https://github.com/frankwuzp/iptv-cn)

## Key Features

*   **Free Live TV:** Access a selection of free Chinese TV channels.
*   **Jellyfin Compatible:** Designed for seamless integration with your Jellyfin media server.
*   **Optimized for China:** Includes optimized `.m3u` and `.xml` files for faster streaming within China.
*   **Automatic EPG Updates:**  `guide.xml` file is automatically updated daily, at 1:00 AM and 6:00 AM, ensuring up-to-date program information.
*   **Multiple Channel Sources:** Provides different `.m3u` sources including those optimized for specific regions and providers.
*   **CDN Support:** Uses jsDelivr CDN for fast and reliable access, especially for users in mainland China.
*   **Easy to Use:** Simple setup instructions for adding the channels to your Jellyfin server.

##  Available Resources

The repository provides `.m3u` and `.xml` files with varying source URLs:

### Channel Lists (.m3u)

*   `tv-ipv4-cn`: General purpose IPTV streams for mainland China.
*   `tv-ipv4-cmcc`: Mobile stream for CMCC Users (Tested working).

### Guide File (EPG - Electronic Program Guide)

*   `guide.xml`: Provides the schedule of programing information.
*   **External EPG Sources:** Additional EPG options are available:
    *   某神秘大神版
    *   iptv-org

## Getting Started

1.  **Choose a Channel List:**
    *   For general use, select `tv-ipv4-cn.m3u`.
    *   For CMCC users, select `tv-ipv4-cmcc.m3u`.
    *   You can use the raw GitHub URL or the jsDelivr CDN URL (recommended for Mainland China users) .

    **Example:**

    *   **Github:** `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cmcc.m3u`
    *   **jsDelivr CDN:** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cmcc.m3u`

2.  **Import into Jellyfin:**
    *   Copy your selected URL (CDN or Github)
    *   Paste the URL into your Jellyfin server TV settings.

    ![jellyfin-setting](./image/jellyfin-settings.jpg)

3.  **Choose an EPG Source:** Choose your preferred EPG source and add it to your Jellyfin setup.

    ![jellyfin-epg](./image/jellyfin-epg.jpg)

## Reference

*   [BurningC4/Chinese-IPTV](https://github.com/BurningC4/Chinese-IPTV)
*   [SoPudge/kodi_iptv_epg](https://github.com/SoPudge/kodi_iptv_epg)
*   [BurningC4/getepg](https://github.com/BurningC4/getepg)
*   [3mile/cctv_api_get_EPG](https://github.com/3mile/cctv_api_get_EPG)
*   [国内高清直播live - TV001](http://www.tv001.vip/forum.php?mod=viewthread&tid=3)
*   [广东移动某河全套 - 恩山无线论坛](https://www.right.com.cn/forum/thread-6809023-1-1.html)

## Changelog

*   **2021-11-26:** Updated channel source and added new mobile signal source.
*   **2021-11-23:** Fixed EPG update issues and added an additional EPG source.
*   **2021-11-22:** Added `guide.xml` EPG file with automated daily updates (1:00 AM & 6:00 AM).  Separated channels into general and Guangdong-specific versions.
*   **2021-11-21:** Initial release.