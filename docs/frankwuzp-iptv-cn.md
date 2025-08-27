# IPTV-CN: Free IPTV Channels for Jellyfin (China)

**Get access to free IPTV channels, specifically curated for use with Jellyfin, including channels optimized for China, with this handy resource.** [View the original repository](https://github.com/frankwuzp/iptv-cn).

[![GitHub last commit (branch)](https://img.shields.io/github/last-commit/frankwuzp/iptv-cn/main?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![GitHub repo size](https://img.shields.io/github/repo-size/frankwuzp/iptv-cn?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![jsdelivr badge](https://data.jsdelivr.com/v1/package/gh/frankwuzp/iptv-cn/badge)](https://www.jsdelivr.com/package/gh/frankwuzp/iptv-cn)
[![GitHub watchers](https://img.shields.io/github/watchers/frankwuzp/iptv-cn?style=social)](https://github.com/frankwuzp/iptv-cn)

## Key Features

*   **Free IPTV Channels:** Access a variety of free IPTV channels.
*   **Optimized for China:** Includes channels specifically for viewers in China.
*   **Jellyfin Compatible:** Designed for easy integration with your Jellyfin media server.
*   **Automatic EPG Updates:**  Electronic Program Guide (EPG) updated automatically twice a day.
*   **Multiple Channel Sources:**  Offers different channel lists, including those optimized for specific regions and providers.
*   **CDN Support:**  Provides CDN links for faster loading in mainland China.

## Available IPTV Channel Lists

The repository provides multiple `.m3u` files containing channel lists.  The following are currently available:

*   `tv-ipv4-cn`: General channel list for mainland China.
*   `tv-ipv4-cmcc`: Channel list, tested and working.
*   `tv-ipv4-old`: Original data, may have issues with lag.

## How to Use

### 1.  Accessing the Channel Lists

You can use the channel lists directly in your Jellyfin setup using the following URLs:

*   **Github (Recommended):**  `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cmcc.m3u`
*   **jsDelivr CDN (for faster loading in mainland China):** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cmcc.m3u`

### 2. Adding to Jellyfin

1.  Save the desired `.m3u` file (e.g., `tv-ipv4-cmcc.m3u`) to your device, or use the direct link.
2.  In your Jellyfin server, go to **Live TV > Add tuner**.
3.  Select **M3U Tuner**.
4.  Enter the URL or the path to your saved `.m3u` file.
5.  Configure the remaining settings as needed.

![jellyfin-settings](./image/jellyfin-settings.jpg)

### 3.  Electronic Program Guide (EPG)

To get program information for your channels, use one of the following EPG source URLs:

*   **某神秘大神版:** `http://epg.51zmt.top:8000/e.xml`
*   **Github:** `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/guide.xml`
*   **jsDelivr CDN (optimized for mainland users):** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn/guide.xml`
*   **iptv-org:** `https://iptv-org.github.io/epg/guides/cn/tv.cctv.com.epg.xml`

![jellyfin-epg](./image/jellyfin-epg.jpg)

## References

*   [BurningC4/Chinese-IPTV](https://github.com/BurningC4/Chinese-IPTV)
*   [SoPudge/kodi_iptv_epg](https://github.com/SoPudge/kodi_iptv_epg)
*   [BurningC4/getepg](https://github.com/BurningC4/getepg)
*   [3mile/cctv_api_get_EPG](https://github.com/3mile/cctv_api_get_EPG)
*   [国内高清直播live - TV001](http://www.tv001.vip/forum.php?mod=viewthread&tid=3)
*   [广东移动某河全套 - 恩山无线论坛](https://www.right.com.cn/forum/thread-6809023-1-1.html)

**Thank you to the open internet for making this possible!**

## Changelog

*   211126: Marked non-functional live streams and added a new mobile source.
*   211123: Fixed issue with deleting old EPG data during updates and added an additional EPG source.
*   211122: Added `guide.xml` EPG file and automated updates (1 AM and 6 AM daily).
*   211122: Separated channels into general and Guangdong-specific versions.
*   211121: Initial release.