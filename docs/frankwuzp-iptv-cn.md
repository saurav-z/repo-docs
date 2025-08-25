# IPTV CN: Free IPTV Resources for Jellyfin (and More!)

**Enjoy free access to Chinese IPTV channels with this repository, specifically optimized for use with Jellyfin and other compatible media platforms!**  Find the original repo [here](https://github.com/frankwuzp/iptv-cn).

[![GitHub last commit](https://img.shields.io/github/last-commit/frankwuzp/iptv-cn/main?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![GitHub repo size](https://img.shields.io/github/repo-size/frankwuzp/iptv-cn?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![jsdelivr package](https://data.jsdelivr.com/v1/package/gh/frankwuzp/iptv-cn/badge)](https://www.jsdelivr.com/package/gh/frankwuzp/iptv-cn)
[![GitHub watchers](https://img.shields.io/github/watchers/frankwuzp/iptv-cn?style=social)](https://github.com/frankwuzp/iptv-cn)

## Key Features

*   **Free IPTV Channels:** Access a variety of Chinese IPTV channels.
*   **Jellyfin Compatibility:** Designed for seamless integration with Jellyfin for live TV streaming.
*   **Optimized for Mainland China:** Includes CDN options for faster loading speeds in mainland China.
*   **Automatic EPG Updates:**  The Electronic Program Guide (EPG) is updated automatically.
*   **Multiple Channel Sources:** Provides different channel list options, including general and mobile-specific sources.
*   **Easy to Use:** Simple setup instructions for Jellyfin and other platforms.

## Channel Lists

This repository provides various `.m3u` channel list files for use with IPTV players like Jellyfin.

### Available Channel Lists:

*   `tv-ipv4-cn`: General Chinese IPTV channels.
*   `tv-ipv4-cmcc`: Channels from China Mobile, tested and working.
*   `tv-ipv4-old`:  Legacy channel list (may have some issues).

### How to Use with Jellyfin:

1.  **Choose a Channel List:** Select a `.m3u` file.  `tv-ipv4-cmcc.m3u` is recommended.
2.  **Get the URL:**
    *   **GitHub:** `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cmcc.m3u`
    *   **jsDelivr CDN (Recommended for Mainland China):** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cmcc.m3u`
3.  **Add to Jellyfin:** In your Jellyfin TV settings, enter the URL of your chosen channel list in the "TV provider" or similar field.
    
    ![jellyfin-setting](./image/jellyfin-settings.jpg)

## Electronic Program Guide (EPG)

An Electronic Program Guide (EPG) provides program information.  This repository provides guide files as well.

### Available EPG Sources:

*   **Secret Source:** `http://epg.51zmt.top:8000/e.xml`
*   **GitHub:** `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/guide.xml`
*   **jsDelivr CDN (Recommended for Mainland China):** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/guide.xml`
*   **iptv-org:** `https://iptv-org.github.io/epg/guides/cn/tv.cctv.com.epg.xml`

    ![jellyfin-epg](./image/jellyfin-epg.jpg)

## References

*   [BurningC4/Chinese-IPTV](https://github.com/BurningC4/Chinese-IPTV)
*   [SoPudge/kodi_iptv_epg](https://github.com/SoPudge/kodi_iptv_epg)
*   [BurningC4/getepg](https://github.com/BurningC4/getepg)
*   [3mile/cctv_api_get_EPG](https://github.com/3mile/cctv_api_get_EPG)
*   [国内高清直播live - TV001](http://www.tv001.vip/forum.php?mod=viewthread&tid=3)
*   [广东移动某河全套 - 恩山无线论坛](https://www.right.com.cn/forum/thread-6809023-1-1.html)

## Changelog

*   211126: Updated notes on unusable sources; added mobile signal source.
*   211123: Resolved EPG update issues and added an additional EPG source.
*   211122: Introduced EPG guide file (`guide.xml`) with automatic updates (1 AM and 6 AM).
*   211122: Separated channel lists into general and Guangdong-specific versions.
*   211121: Initial release.