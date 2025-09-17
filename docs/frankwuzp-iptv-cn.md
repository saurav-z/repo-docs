# IPTV-CN: Free Live TV Channels for Jellyfin (China)

**Looking to watch free live TV channels in China with Jellyfin?** IPTV-CN provides updated M3U and EPG files for a seamless viewing experience.  [View the original repository](https://github.com/frankwuzp/iptv-cn).

[![GitHub last commit (branch)](https://img.shields.io/github/last-commit/frankwuzp/iptv-cn/main?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![GitHub repo size](https://img.shields.io/github/repo-size/frankwuzp/iptv-cn?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![jsdelivr](https://data.jsdelivr.com/v1/package/gh/frankwuzp/iptv-cn/badge)](https://www.jsdelivr.com/package/gh/frankwuzp/iptv-cn)
[![GitHub watchers](https://img.shields.io/github/watchers/frankwuzp/iptv-cn?style=social)](https://github.com/frankwuzp/iptv-cn)

## Key Features:

*   **Updated M3U Files:**  Provides the latest working IPTV channel lists, including a China-wide general list and a Mobile list.
*   **EPG (Electronic Program Guide):** Includes a regularly updated `guide.xml` file for TV program listings.
*   **Jellyfin Compatibility:** Designed specifically for use with Jellyfin's live TV functionality.
*   **CDN Acceleration:** Uses jsDelivr CDN for faster access, particularly for users in mainland China.
*   **Automated Updates:** EPG data is automatically updated daily at 1 AM and 6 AM via GitHub Actions.

## How to Use:

This repository provides the necessary files to integrate live TV channels into your Jellyfin setup.

### Files Explained:

*   `tv-ipv4-cn.m3u`: General IPTV playlist for China (Universal).
*   `tv-ipv4-cmcc.m3u`:  Mobile IPTV playlist (China Mobile) - Recommended.
*   `tv-ipv4-old.m3u`: Original data from the Chinese-IPTV repo.
*   `guide.xml`: Electronic Program Guide (EPG) data, automatically updated daily.
*   `requirements.txt`:  Python dependencies for the `get-epg.py` script (used for generating the EPG).

### Setting up Channel Lists in Jellyfin:

1.  **Choose a Channel List Source:**
    *   **GitHub:**  `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cmcc.m3u` (Recommended, most up-to-date)
    *   **jsDelivr CDN:** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cmcc.m3u` (Faster access in China)
    *   Save the `tv-ipv4-cmcc.m3u` file from the repository.

2.  **Enter the URL in Jellyfin:**  In your Jellyfin Live TV settings, enter the chosen URL (or the path to the saved file) into the M3U URL field.

    ![jellyfin-setting](./image/jellyfin-settings.jpg)

### Setting up the Electronic Program Guide (EPG) in Jellyfin:

1.  **Choose an EPG Source:**
    *   **某神秘大神版**: `http://epg.51zmt.top:8000/e.xml` (Recommend, most up-to-date)
    *   **GitHub:** `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/guide.xml`
    *   **jsDelivr CDN (China-optimized):** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/guide.xml`
    *   **iptv-org**: `https://iptv-org.github.io/epg/guides/cn/tv.cctv.com.epg.xml`

2.  **Enter the EPG URL in Jellyfin:**  In your Jellyfin Live TV settings, enter the chosen EPG URL.

    ![jellyfin-epg](./image/jellyfin-epg.jpg)

## References:

*   [BurningC4/Chinese-IPTV](https://github.com/BurningC4/Chinese-IPTV)
*   [SoPudge/kodi_iptv_epg](https://github.com/SoPudge/kodi_iptv_epg)
*   [BurningC4/getepg](https://github.com/BurningC4/getepg)
*   [3mile/cctv_api_get_EPG](https://github.com/3mile/cctv_api_get_EPG)
*   [国内高清直播live - TV001](http://www.tv001.vip/forum.php?mod=viewthread&tid=3)
*   [广东移动某河全套 - 恩山无线论坛](https://www.right.com.cn/forum/thread-6809023-1-1.html)

**Thanks to the Open Internet! 🎉🎉🎉**

## Changelog:

*   211126: Added mobile signal source.
*   211123: Fixed EPG update issues and added a new EPG source.
*   211122: Added EPG file (`guide.xml`) and automated daily updates.
*   211122: Separated into general and Guangdong-specific versions.
*   211121: Initial commit.