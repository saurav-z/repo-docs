# IPTV CN: Free Live TV for Jellyfin in China (and Guangdong)

Easily access and enjoy live TV channels in China, optimized for use with Jellyfin, with this repository.  **[View the original repo on GitHub](https://github.com/frankwuzp/iptv-cn)**.

[![GitHub last commit (branch)](https://img.shields.io/github/last-commit/frankwuzp/iptv-cn/main?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![GitHub repo size](https://img.shields.io/github/repo-size/frankwuzp/iptv-cn?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![jsdelivr](https://data.jsdelivr.com/v1/package/gh/frankwuzp/iptv-cn/badge)](https://www.jsdelivr.com/package/gh/frankwuzp/iptv-cn)
[![GitHub watchers](https://img.shields.io/github/watchers/frankwuzp/iptv-cn?style=social)](https://github.com/frankwuzp/iptv-cn)


## Key Features

*   **Free IPTV Resources:** Provides M3U playlists and EPG (Electronic Program Guide) files for live TV channels in China.
*   **Jellyfin Compatibility:** Designed for seamless integration with Jellyfin for live TV streaming.
*   **Optimized for Mainland China:** Includes CDN options (jsDelivr) for faster access in China.
*   **Automatic EPG Updates:**  `guide.xml` file automatically updates daily at 1 AM and 6 AM to ensure accurate program information.
*   **Multiple Channel Source Options:** Offers various channel lists, including ones optimized for Guangdong province and China Mobile users.

## Getting Started

This repository provides the necessary resources to set up live TV in Jellyfin.

### Files Explained

*   `tv-ipv4-cn.m3u`:  General-purpose M3U file for channels across China.
*   `tv-ipv4-cmcc.m3u`: Optimized channel list for China Mobile users (tested and working).
*   `tv-ipv4-old.m3u`:  Older channel list (may have latency or be partially unavailable).  Based on [BurningC4/Chinese-IPTV](https://github.com/BurningC4/Chinese-IPTV).
*   `guide.xml`:  EPG file generated daily via automated Actions (at 1 AM and 6 AM).
*   `requirements.txt`: Python dependencies for the `get-epg.py` script.

### Using the Channel Lists (M3U)

You can obtain the M3U channel list file from one of the following sources and add the URL to Jellyfin:

*   **GitHub:** `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cmcc.m3u`
*   **jsDelivr CDN (Recommended for Mainland China):** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cmcc.m3u`

**How to add to Jellyfin:**
1.  Select the URL
2.  Paste it into the "Network URL" or similar field in your Jellyfin TV Live settings.
3.  Save the configuration and start enjoying live tv!

![jellyfin-setting](./image/jellyfin-settings.jpg)

### Using the Electronic Program Guide (EPG)

You can add the EPG to Jellyfin as well. Use one of the URLs below.

*   **某神秘大神版**

  `http://epg.51zmt.top:8000/e.xml`

*   **Github**

  `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/guide.xml`

*   **jsDelivr CDN (optimized for mainland users)**

  `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/guide.xml`

*   **iptv-org**
  
  `https://iptv-org.github.io/epg/guides/cn/tv.cctv.com.epg.xml`

![jellyfin-epg](./image/jellyfin-epg.jpg)

## References

*   [BurningC4/Chinese-IPTV](https://github.com/BurningC4/Chinese-IPTV)
*   [SoPudge/kodi_iptv_epg](https://github.com/SoPudge/kodi_iptv_epg)
*   [BurningC4/getepg](https://github.com/BurningC4/getepg)
*   [3mile/cctv_api_get_EPG](https://github.com/3mile/cctv_api_get_EPG)
*   [国内高清直播live - TV001](http://www.tv001.vip/forum.php?mod=viewthread&tid=3)
*   [广东移动某河全套 - 恩山无线论坛](https://www.right.com.cn/forum/thread-6809023-1-1.html)

## Changelog

*   211126:  Added note about unavailable sources and introduced the China Mobile channel source.
*   211123: Resolved issues with deleting old content during EPG updates and added a new EPG source.
*   211122:  Added `guide.xml` EPG file and automated updates (1 AM & 6 AM).
*   211122: Introduced separate versions for general use and Guangdong province.
*   211121: Initial Release.

**Thanks to the open internet! 🎉🎉🎉**