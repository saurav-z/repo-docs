# IPTV-CN: Watch Chinese IPTV Channels with Jellyfin

**Access and enjoy live Chinese TV channels with ease using this regularly updated IPTV resource, optimized for use with Jellyfin.**  [View the original repository on GitHub](https://github.com/frankwuzp/iptv-cn)

[![GitHub last commit (branch)](https://img.shields.io/github/last-commit/frankwuzp/iptv-cn/main?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![GitHub repo size](https://img.shields.io/github/repo-size/frankwuzp/iptv-cn?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![jsdelivr](https://data.jsdelivr.com/v1/package/gh/frankwuzp/iptv-cn/badge)](https://www.jsdelivr.com/package/gh/frankwuzp/iptv-cn)
[![GitHub watchers](https://img.shields.io/github/watchers/frankwuzp/iptv-cn?style=social)](https://github.com/frankwuzp/iptv-cn)

## Key Features:

*   **Reliable IPTV Resources:** Provides working IPTV resources for watching Chinese TV channels.
*   **Jellyfin Compatibility:** Designed for seamless integration with the Jellyfin media server.
*   **Region-Specific Channels:** Includes optimized channel lists, with a focus on Guangdong province.
*   **Automatic EPG Updates:**  The Electronic Program Guide (EPG) is automatically updated daily at 1:00 AM and 6:00 AM.
*   **Multiple Source Options:** Offers channel lists and EPG files from GitHub and jsDelivr CDN for faster access.
*   **Mobile Channel Source**: Includes mobile channel source for more streaming options

## How to Use:

### Files Explained:

*   `tv-ipv4-cn.m3u`: General IPTV channel list for China.
*   `tv-ipv4-cmcc.m3u`: Updated channel list for mobile, tested working.
*   `tv-ipv4-old.m3u`:  Older channel list (may have some delays or issues).
*   `guide.xml`: Electronic Program Guide (EPG) file, automatically updated.
*   `requirements.txt`: Python script dependencies for generating the EPG (`get-epg.py`).

### Setting Up in Jellyfin:

1.  **Channel List:**
    *   **Recommended:** Use `tv-ipv4-cmcc.m3u` for mobile.
    *   **Direct Link (GitHub):** `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cmcc.m3u`
    *   **CDN Link (jsDelivr):** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cmcc.m3u`
    *   Copy the URL of your preferred channel list to your Jellyfin live TV settings (see image below).
2.  **EPG Guide File:**
    *   **Direct Link (GitHub):** `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/guide.xml`
    *   **CDN Link (jsDelivr):** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/guide.xml`
    *   **Alternative Sources:**
        *   `http://epg.51zmt.top:8000/e.xml` (Mysterious God Version)
        *   `https://iptv-org.github.io/epg/guides/cn/tv.cctv.com.epg.xml`
    *   Copy the URL of your EPG file to your Jellyfin live TV settings (see image below).

    ![jellyfin-setting](./image/jellyfin-settings.jpg)

    ![jellyfin-epg](./image/jellyfin-epg.jpg)

## References:

*   [BurningC4/Chinese-IPTV](https://github.com/BurningC4/Chinese-IPTV)
*   [SoPudge/kodi_iptv_epg](https://github.com/SoPudge/kodi_iptv_epg)
*   [BurningC4/getepg](https://github.com/BurningC4/getepg)
*   [3mile/cctv_api_get_EPG](https://github.com/3mile/cctv_api_get_EPG)
*   [国内高清直播live - TV001](http://www.tv001.vip/forum.php?mod=viewthread&tid=3)
*   [广东移动某河全套 - 恩山无线论坛](https://www.right.com.cn/forum/thread-6809023-1-1.html)

## Changelog:

*   **2021-11-26:** Added mobile channel source.
*   **2021-11-23:** Improved EPG updates.
*   **2021-11-22:** Added EPG guide file and automatic updates.
*   **2021-11-22:** Split into general and Guangdong-specific versions.
*   **2021-11-21:** Initial Release.