# IPTV China: Free IPTV Channels for Jellyfin and More

Easily access and stream live TV channels in China, particularly optimized for Guangdong province, with this free and regularly updated IPTV resource.  Check out the [original repo](https://github.com/frankwuzp/iptv-cn) for the latest updates and information.

[![GitHub last commit](https://img.shields.io/github/last-commit/frankwuzp/iptv-cn/main?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![GitHub repo size](https://img.shields.io/github/repo-size/frankwuzp/iptv-cn?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![jsdelivr](https://data.jsdelivr.com/v1/package/gh/frankwuzp/iptv-cn/badge)](https://www.jsdelivr.com/package/gh/frankwuzp/iptv-cn)
[![GitHub watchers](https://img.shields.io/github/watchers/frankwuzp/iptv-cn?style=social)](https://github.com/frankwuzp/iptv-cn)

## Key Features

*   **Free IPTV Channels:** Access a variety of Chinese TV channels without subscription fees.
*   **Optimized for Guangdong:** Specifically designed for users in Guangdong province, with channels optimized for local access.
*   **Jellyfin Compatible:** Easy integration with Jellyfin for seamless live TV streaming.
*   **Regularly Updated:** Channels and EPG (Electronic Program Guide) data are frequently updated for reliability.
*   **Multiple Source Options:**  Includes both Github and CDN (jsDelivr) links for channel lists.
*   **Automated EPG Updates:** The EPG is automatically generated and updated daily.

## Available Files

*   `tv-ipv4-cn.m3u`: General-purpose M3U file for IPTV channels within China.
*   `tv-ipv4-cmcc.m3u`:  Optimized for China Mobile users (tested and confirmed working).
*   `tv-ipv4-old.m3u`:  Archive of older channel data (may have some issues with latency or stability).
*   `guide.xml`:  EPG (Electronic Program Guide) data, automatically updated daily.
*   `requirements.txt`:  Python script dependencies for generating the EPG.

## How to Use

### Channel Lists

You can use the following M3U links in your IPTV player (e.g., Jellyfin):

*   **Github:**  `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cmcc.m3u`
*   **jsDelivr CDN (Recommended for Mainland China):** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cmcc.m3u`

  **Tip:** Consider saving the `.m3u` file locally for offline use.

### EPG (Electronic Program Guide)

Choose one of the following EPG source options:

*   **Github:** `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/guide.xml`
*   **jsDelivr CDN (Optimized for Mainland China):**  `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/guide.xml`
*   **iptv-org:** `https://iptv-org.github.io/epg/guides/cn/tv.cctv.com.epg.xml`
*   **External Source:**  `http://epg.51zmt.top:8000/e.xml`

Insert the EPG URL into your IPTV player's settings (example shown below for Jellyfin):

![jellyfin-epg](./image/jellyfin-epg.jpg)

## References

*   [BurningC4/Chinese-IPTV](https://github.com/BurningC4/Chinese-IPTV)
*   [SoPudge/kodi_iptv_epg](https://github.com/SoPudge/kodi_iptv_epg)
*   [BurningC4/getepg](https://github.com/BurningC4/getepg)
*   [3mile/cctv_api_get_EPG](https://github.com/3mile/cctv_api_get_EPG)
*   [国内高清直播live - TV001](http://www.tv001.vip/forum.php?mod=viewthread&tid=3)
*   [广东移动某河全套 - 恩山无线论坛](https://www.right.com.cn/forum/thread-6809023-1-1.html)

## Changelog

*   211126: Updated channel source availability, added China Mobile source.
*   211123: Resolved EPG update issues, added a new EPG source.
*   211122: Implemented automatic EPG updates (daily at 1 AM and 6 AM), and introduced separate channel lists for general use and Guangdong.
*   211121: Initial release.

**Thank you to the open internet for making this possible!** 🎉