# IPTV-CN: Watch Chinese TV Channels with Jellyfin

**Looking to stream Chinese TV channels on Jellyfin?** IPTV-CN provides up-to-date M3U playlists and EPG data for a seamless viewing experience, with a focus on reliable sources. ([View on GitHub](https://github.com/frankwuzp/iptv-cn))

[![GitHub last commit (branch)](https://img.shields.io/github/last-commit/frankwuzp/iptv-cn/main?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![GitHub repo size](https://img.shields.io/github/repo-size/frankwuzp/iptv-cn?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![jsdelivr](https://data.jsdelivr.com/v1/package/gh/frankwuzp/iptv-cn/badge)](https://www.jsdelivr.com/package/gh/frankwuzp/iptv-cn)
[![GitHub watchers](https://img.shields.io/github/watchers/frankwuzp/iptv-cn?style=social)](https://github.com/frankwuzp/iptv-cn)

## Key Features

*   **Reliable M3U Playlists:** Provides tested IPTV resources, including a specifically optimized playlist for China Mobile users.
*   **Jellyfin Compatibility:** Designed for seamless integration with Jellyfin's live TV functionality.
*   **Automatic EPG Updates:** The Electronic Program Guide (`guide.xml`) is automatically updated daily via GitHub Actions.
*   **CDN Acceleration:** Offers jsDelivr CDN links for faster access, especially for users in Mainland China.
*   **Multiple Channel Source Options:** Includes both general and carrier-specific channel lists for flexibility.

## How to Use

### M3U Playlist Files

This repository offers several M3U playlist files, with recommendations for optimal viewing:

*   `tv-ipv4-cn.m3u`: General-purpose IPTV playlist (Recommended)
*   `tv-ipv4-cmcc.m3u`: Optimized playlist for China Mobile users (Recommended)
*   `tv-ipv4-old.m3u`: Older, potentially less reliable sources.

**To use the playlists in Jellyfin:**

1.  Choose an M3U file (e.g., `tv-ipv4-cmcc.m3u`).
2.  Use the Github link: `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cmcc.m3u`
3.  **Or,** use the jsDelivr CDN link for faster access: `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cmcc.m3u`
4.  Enter the URL into your Jellyfin TV tuner settings.

![jellyfin-setting](./image/jellyfin-settings.jpg)

### EPG (Electronic Program Guide) Files

The EPG provides program information for your channels.  Choose one of the following sources:

*   **某神秘大神版**: `http://epg.51zmt.top:8000/e.xml`
*   **GitHub**: `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/guide.xml`
*   **jsDelivr CDN (Optimized for Mainland China)**: `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/guide.xml`
*   **iptv-org**: `https://iptv-org.github.io/epg/guides/cn/tv.cctv.com.epg.xml`

![jellyfin-epg](./image/jellyfin-epg.jpg)

## References

*   [BurningC4/Chinese-IPTV](https://github.com/BurningC4/Chinese-IPTV)
*   [SoPudge/kodi_iptv_epg](https://github.com/SoPudge/kodi_iptv_epg)
*   [BurningC4/getepg](https://github.com/BurningC4/getepg)
*   [3mile/cctv_api_get_EPG](https://github.com/3mile/cctv_api_get_EPG)
*   [国内高清直播live - TV001](http://www.tv001.vip/forum.php?mod=viewthread&tid=3)
*   [广东移动某河全套 - 恩山无线论坛](https://www.right.com.cn/forum/thread-6809023-1-1.html)

## Changelog

*   **2021-11-26:** Updated playlist sources, including the addition of a China Mobile source.
*   **2021-11-23:** Fixed EPG update issues and added a new EPG source.
*   **2021-11-22:** Added automatic EPG updates (daily at 1 AM and 6 AM UTC) and separated playlists into general and Guangdong-specific versions.
*   **2021-11-21:** Initial release.