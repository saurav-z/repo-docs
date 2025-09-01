# IPTV-CN: Free Chinese IPTV Resources for Jellyfin

**Enjoy free Chinese TV channels with this repository, providing IPTV resources compatible with Jellyfin for seamless streaming.** ([View the original repository](https://github.com/frankwuzp/iptv-cn))

[![GitHub last commit (branch)](https://img.shields.io/github/last-commit/frankwuzp/iptv-cn/main?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![GitHub repo size](https://img.shields.io/github/repo-size/frankwuzp/iptv-cn?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![jsdelivr](https://data.jsdelivr.com/v1/package/gh/frankwuzp/iptv-cn/badge)](https://www.jsdelivr.com/package/gh/frankwuzp/iptv-cn)
[![GitHub watchers](https://img.shields.io/github/watchers/frankwuzp/iptv-cn?style=social)](https://github.com/frankwuzp/iptv-cn)

## Key Features

*   **Free Chinese IPTV Channels:** Access a curated list of free Chinese TV channels.
*   **Jellyfin Compatibility:** Designed for easy integration with Jellyfin for live TV streaming.
*   **Updated Resources:** Regularly updated `m3u` playlist files.
*   **EPG (Electronic Program Guide):** Includes an EPG file (`guide.xml`) for program information, automatically updated daily.
*   **Multiple Source Options:** Offers various sources for both `m3u` and `guide.xml` files, including GitHub and jsDelivr CDN for faster access.
*   **Mobile-Friendly:** Includes dedicated mobile signal sources.

## How to Use

### IPTV Playlist Files (`m3u`)

This repository provides different `m3u` playlist files for various scenarios:

*   `tv-ipv4-cn`: General-purpose IPTV list for mainland China.
*   `tv-ipv4-cmcc`: IPTV list specifically for mobile users, tested and working.

To use the playlist in Jellyfin, you can use the following links:

*   **GitHub (raw):** `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cmcc.m3u`
*   **jsDelivr CDN (Recommended for Mainland China users):** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cmcc.m3u`

1.  **Choose a Source:** Select either the GitHub or jsDelivr CDN link.
2.  **Add to Jellyfin:** In your Jellyfin server, navigate to Live TV and configure a new TV tuner using the selected URL.

### Electronic Program Guide (EPG) Files

An EPG file (`guide.xml`) provides program information.  You can use one of the following sources:

*   **Mysterious Provider:** `http://epg.51zmt.top:8000/e.xml`
*   **GitHub (raw):** `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/guide.xml`
*   **jsDelivr CDN (Recommended for Mainland China users):** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/guide.xml`
*   **iptv-org:** `https://iptv-org.github.io/epg/guides/cn/tv.cctv.com.epg.xml`

1.  **Choose a Source:** Select your preferred EPG source.
2.  **Add to Jellyfin:** Configure the EPG source within Jellyfin's Live TV settings.

![jellyfin-settings](./image/jellyfin-settings.jpg)
![jellyfin-epg](./image/jellyfin-epg.jpg)

## References

*   [BurningC4/Chinese-IPTV](https://github.com/BurningC4/Chinese-IPTV)
*   [SoPudge/kodi_iptv_epg](https://github.com/SoPudge/kodi_iptv_epg)
*   [BurningC4/getepg](https://github.com/BurningC4/getepg)
*   [3mile/cctv_api_get_EPG](https://github.com/3mile/cctv_api_get_EPG)
*   [国内高清直播live - TV001](http://www.tv001.vip/forum.php?mod=viewthread&tid=3)
*   [广东移动某河全套 - 恩山无线论坛](https://www.right.com.cn/forum/thread-6809023-1-1.html)

**Thanks to the open Internet! 🎉🎉🎉**

## Changelog

*   **211126:** Updated with mobile signal sources; noted non-working sources.
*   **211123:** Fixed EPG update issues and added an additional EPG source.
*   **211122:** Added automatic EPG update (daily at 1 AM and 6 AM) and the `guide.xml` file.
*   **211122:** Playlist files separated into general and Guangdong-specific versions.
*   **211121:** Initial release.