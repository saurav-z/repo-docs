# IPTV-CN: Watch Chinese TV Channels on Jellyfin (and More!)

Access and stream a variety of Chinese IPTV channels directly within your Jellyfin media server with this easy-to-use resource. ([Original Repo](https://github.com/frankwuzp/iptv-cn))

[![GitHub last commit (branch)](https://img.shields.io/github/last-commit/frankwuzp/iptv-cn/main?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![GitHub repo size](https://img.shields.io/github/repo-size/frankwuzp/iptv-cn?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![jsdelivr](https://data.jsdelivr.com/v1/package/gh/frankwuzp/iptv-cn/badge)](https://www.jsdelivr.com/package/gh/frankwuzp/iptv-cn)
[![GitHub watchers](https://img.shields.io/github/watchers/frankwuzp/iptv-cn?style=social)](https://github.com/frankwuzp/iptv-cn)

## Key Features

*   **Reliable IPTV Resources:** Provides access to working IPTV channels, tested and optimized for use.
*   **Jellyfin Compatibility:**  Designed for seamless integration with Jellyfin for live TV streaming.
*   **Multiple Channel Lists:** Offers several `.m3u` files for different channel sources, including China Mobile (CMCC) and general Chinese channels.
*   **Automatic EPG Updates:** Includes a guide.xml file that automatically updates with Electronic Program Guide (EPG) information at 1 AM and 6 AM daily, making your viewing experience better.
*   **CDN Acceleration:** Uses jsDelivr CDN for faster access, particularly for users in mainland China.

## How to Use

### Files Explained

*   `tv-ipv4-cn.m3u`: General-purpose `.m3u` file for channels across China.
*   `tv-ipv4-cmcc.m3u`: Updated mobile channel list.
*   `tv-ipv4-old.m3u`: Original channel list (may have some issues).
*   `guide.xml`:  EPG file, automatically updated daily.
*   `requirements.txt`: Python dependencies for the `get-epg.py` script.

### Channel Lists (Example: Using CMCC)

You can directly use the following URL (choose one) in your Jellyfin TV tuner configuration:

*   **GitHub:** `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cmcc.m3u`
*   **jsDelivr CDN (Recommended for China):** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cmcc.m3u`

  *   Download the `tv-ipv4-cmcc.m3u` file and point Jellyfin to the downloaded file, if you wish.
  *   Configure Jellyfin TV tuner

### EPG Guide File (Choose One)

*   **Github**

    `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/guide.xml`

*   **jsDelivr CDN (optimized for mainland users)**

    `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/guide.xml`

*   **iptv-org**

    `https://iptv-org.github.io/epg/guides/cn/tv.cctv.com.epg.xml`

  *   Configure Jellyfin EPG

## Changelog

*   211126: Added mobile channel source, updated notes on unusable sources.
*   211123: Fixed EPG update issues, added a new EPG source.
*   211122: Added `guide.xml` for automatic EPG updates.
*   211122:  Separated into general and Guangdong-specific versions.
*   211121: Initial release.

## References

*   [BurningC4/Chinese-IPTV](https://github.com/BurningC4/Chinese-IPTV)
*   [SoPudge/kodi\_iptv\_epg](https://github.com/SoPudge/kodi_iptv_epg)
*   [BurningC4/getepg](https://github.com/BurningC4/getepg)
*   [3mile/cctv\_api\_get\_EPG](https://github.com/3mile/cctv_api_get_EPG)
*   [国内高清直播live - TV001](http://www.tv001.vip/forum.php?mod=viewthread&tid=3)
*   [广东移动某河全套 - 恩山无线论坛](https://www.right.com.cn/forum/thread-6809023-1-1.html)

**Enjoy your Chinese TV streaming!**