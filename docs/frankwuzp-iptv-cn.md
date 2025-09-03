# IPTV-CN: Watch Chinese TV Channels with Jellyfin

Tired of missing your favorite Chinese TV shows? This repository provides up-to-date IPTV resources, specifically designed for use with Jellyfin, making it easy to stream live Chinese TV channels. ([Back to original repo](https://github.com/frankwuzp/iptv-cn))

[![GitHub last commit (branch)](https://img.shields.io/github/last-commit/frankwuzp/iptv-cn/main?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![GitHub repo size](https://img.shields.io/github/repo-size/frankwuzp/iptv-cn?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![jsdelivr](https://data.jsdelivr.com/v1/package/gh/frankwuzp/iptv-cn/badge)](https://www.jsdelivr.com/package/gh/frankwuzp/iptv-cn)
[![GitHub watchers](https://img.shields.io/github/watchers/frankwuzp/iptv-cn?style=social)](https://github.com/frankwuzp/iptv-cn)

## Key Features

*   **Reliable IPTV Resources:** Provides tested and working IPTV streams for Chinese TV channels, including a dedicated source for Guangdong province (and general CN channels).
*   **Optimized for Jellyfin:**  Designed for seamless integration with Jellyfin's live TV functionality.
*   **Automatic EPG Updates:**  Includes an Electronic Program Guide (EPG) file that is automatically updated daily, ensuring you have the latest program information.
*   **Multiple Source Options:** Offers channel lists and EPG sources from Github, jsDelivr CDN (for faster access in mainland China), and other reliable sources.
*   **Easy Setup:** Clear instructions and pre-configured files make it simple to set up live TV within Jellyfin.
*   **Updated Mobile Channels:** Includes updated channels for mobile users.

## How to Use

### Files Explained

*   `tv-ipv4-cn.m3u`: Universal m3u file for channels in China.
*   `tv-ipv4-cmcc.m3u`: Optimized mobile channel source (tested and working).
*   `tv-ipv4-old.m3u`: Source from the `BurningC4/Chinese-IPTV` repository (some channels may be outdated).
*   `guide.xml`: Electronic Program Guide (EPG) file, automatically updated daily.
*   `requirements.txt`: Python dependency file for `get-epg.py`.

### Channel Lists

You can use the following m3u channel list URLs in your Jellyfin setup:

*   **Github:**

    `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cmcc.m3u`

*   **jsDelivr CDN (Recommended for Mainland China):**

    `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cmcc.m3u`

    Either save the `tv-ipv4-cmcc.m3u` file to your device or use the above URLs directly within your Jellyfin TV settings.

    ![jellyfin-setting](./image/jellyfin-settings.jpg)

### EPG Guide Files

Select an EPG guide file for program information:

*   **External:**

    `http://epg.51zmt.top:8000/e.xml`

*   **Github:**

    `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/guide.xml`

*   **jsDelivr CDN (Optimized for Mainland China):**

    `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/guide.xml`

*   **iptv-org:**

    `https://iptv-org.github.io/epg/guides/cn/tv.cctv.com.epg.xml`

    ![jellyfin-epg](./image/jellyfin-epg.jpg)

## Changelog

*   **211126:** Updated channel lists and added mobile source.
*   **211123:** Fixed issue with EPG updates and added an additional EPG source.
*   **211122:** Added EPG guide file (`guide.xml`) with automatic daily updates.
*   **211122:** Separated channel lists into general and Guangdong-specific versions.
*   **211121:** Initial release.

## References

*   [BurningC4/Chinese-IPTV](https://github.com/BurningC4/Chinese-IPTV)
*   [SoPudge/kodi_iptv_epg](https://github.com/SoPudge/kodi_iptv_epg)
*   [BurningC4/getepg](https://github.com/BurningC4/getepg)
*   [3mile/cctv_api_get_EPG](https://github.com/3mile/cctv_api_get_EPG)
*   [国内高清直播live - TV001](http://www.tv001.vip/forum.php?mod=viewthread&tid=3)
*   [广东移动某河全套 - 恩山无线论坛](https://www.right.com.cn/forum/thread-6809023-1-1.html)

**Thanks to the open internet for these resources! 🎉🎉🎉**