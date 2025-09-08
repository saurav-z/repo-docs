# Watch Chinese IPTV with Jellyfin: Get Live TV Channels Easily

This repository provides ready-to-use IPTV resources specifically for Chinese TV channels, optimized for streaming on Jellyfin.  Find the original repository at [frankwuzp/iptv-cn](https://github.com/frankwuzp/iptv-cn).

[![GitHub last commit (branch)](https://img.shields.io/github/last-commit/frankwuzp/iptv-cn/main?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![GitHub repo size](https://img.shields.io/github/repo-size/frankwuzp/iptv-cn?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![jsdelivr](https://data.jsdelivr.com/v1/package/gh/frankwuzp/iptv-cn/badge)](https://www.jsdelivr.com/package/gh/frankwuzp/iptv-cn)
[![GitHub watchers](https://img.shields.io/github/watchers/frankwuzp/iptv-cn?style=social)](https://github.com/frankwuzp/iptv-cn)

## Key Features:

*   **Ready-to-Use M3U Files:** Easily integrate live TV channels into your Jellyfin setup.
*   **Optimized for China:** Includes channel lists suitable for mainland China, with an emphasis on Guangdong province.
*   **Updated Channel Sources:**  Provides multiple sources, including those optimized for China Mobile users.
*   **Automatic EPG Updates:**  Enjoy up-to-date Electronic Program Guide (EPG) data, updated daily.
*   **CDN Support:**  Offers jsDelivr CDN links for faster streaming, especially for users in mainland China.
*   **Jellyfin Integration:** Detailed instructions and screenshots for seamless integration with Jellyfin.

## Available Files:

*   `tv-ipv4-cn`: General purpose IPTV channels for mainland China.
*   `tv-ipv4-cmcc`: IPTV channels specifically for China Mobile users.
*   `tv-ipv4-old`:  Archive of older IPTV sources (may have some limitations).
*   `guide.xml`: Electronic Program Guide (EPG) data, automatically updated daily at 1 AM and 6 AM (UTC+8).

## How to Use:

### Channel Lists (M3U)

Choose one of the following sources and enter it as the "Network location" in your Jellyfin Live TV settings.  We recommend using `tv-ipv4-cmcc.m3u` for users of China Mobile.

*   **GitHub (Raw):** `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cmcc.m3u`
*   **jsDelivr CDN (Optimized):** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cmcc.m3u`

   ![Jellyfin Settings Example](./image/jellyfin-settings.jpg)

### Guide File (EPG - Electronic Program Guide)

Select one of the following sources and enter it as the EPG source in your Jellyfin Live TV settings.

*   **GitHub (Raw):**  `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/guide.xml`
*   **jsDelivr CDN (Optimized):** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/guide.xml`
*   **iptv-org:** `https://iptv-org.github.io/epg/guides/cn/tv.cctv.com.epg.xml`

   ![Jellyfin EPG Example](./image/jellyfin-epg.jpg)

## References:

This project leverages and is inspired by the following resources:

*   [BurningC4/Chinese-IPTV](https://github.com/BurningC4/Chinese-IPTV)
*   [SoPudge/kodi_iptv_epg](https://github.com/SoPudge/kodi_iptv_epg)
*   [BurningC4/getepg](https://github.com/BurningC4/getepg)
*   [3mile/cctv_api_get_EPG](https://github.com/3mile/cctv_api_get_EPG)
*   [国内高清直播live - TV001](http://www.tv001.vip/forum.php?mod=viewthread&tid=3)
*   [广东移动某河全套 - 恩山无线论坛](https://www.right.com.cn/forum/thread-6809023-1-1.html)

## Changelog:

*   **2021-11-26:** Added China Mobile channel source and updated channel availability notes.
*   **2021-11-23:** Fixed EPG update issues and added a new EPG source.
*   **2021-11-22:** Implemented automatic EPG updates and introduced a distinction between general and Guangdong-specific channel lists.
*   **2021-11-22:** Added EPG guide file `guide.xml` and automatic updates (daily at 1 AM and 6 AM).
*   **2021-11-21:** Initial Release.