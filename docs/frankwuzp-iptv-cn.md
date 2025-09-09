# IPTV CN: Watch Chinese TV Channels with Jellyfin

**Access and enjoy Chinese IPTV channels seamlessly with this repository, optimized for Jellyfin.**  ([Original Repository](https://github.com/frankwuzp/iptv-cn))

[![GitHub last commit](https://img.shields.io/github/last-commit/frankwuzp/iptv-cn/main?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![GitHub repo size](https://img.shields.io/github/repo-size/frankwuzp/iptv-cn?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![jsdelivr](https://data.jsdelivr.com/v1/package/gh/frankwuzp/iptv-cn/badge)](https://www.jsdelivr.com/package/gh/frankwuzp/iptv-cn)
[![GitHub watchers](https://img.shields.io/github/watchers/frankwuzp/iptv-cn?style=social)](https://github.com/frankwuzp/iptv-cn)

## Key Features

*   **Chinese IPTV Channels:** Provides access to Chinese TV channels.
*   **Jellyfin Compatibility:** Specifically designed for use with Jellyfin media server.
*   **Updated Resources:** Includes regularly updated M3U playlists and EPG (Electronic Program Guide) data.
*   **Optimized for Mainland Users:** Offers CDN (Content Delivery Network) options for faster streaming in China.
*   **Automatic EPG Updates:** The `guide.xml` file is automatically updated daily.
*   **Multiple Source Options:** Choose from various channel list sources and EPG providers.

## Getting Started

### Files Explained

*   `tv-ipv4-cn.m3u`: General IPTV playlist for China.
*   `tv-ipv4-cmcc.m3u`: IPTV playlist optimized for China Mobile users (tested and working).
*   `tv-ipv4-old.m3u`:  An older playlist (from BurningC4/Chinese-IPTV), some channels may still work but with potential latency.
*   `guide.xml`: EPG (Electronic Program Guide) file; updated daily via GitHub Actions.
*   `requirements.txt`: Dependencies for the `get-epg.py` script.

### How to Use with Jellyfin

1.  **Choose a Channel List:** Select a channel list (`.m3u` file).  The recommended option is `tv-ipv4-cmcc.m3u`.
    *   **Github:**  `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cmcc.m3u`
    *   **jsDelivr CDN (Recommended for China):**  `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cmcc.m3u`
    *   You can download the `.m3u` file or use the URLs directly in Jellyfin.

2.  **Configure Jellyfin:** In your Jellyfin server, add the chosen `.m3u` URL as a TV tuner.

    ![jellyfin-setting](./image/jellyfin-settings.jpg)

3.  **Choose an EPG Guide:** Select an EPG source to populate your TV guide.
    *   **Recommended:**
        *   **jsDelivr CDN (Optimized for mainland users):**  `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/guide.xml`
    *   Other Options:
        *   **Github:**  `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/guide.xml`
        *   **iptv-org:** `https://iptv-org.github.io/epg/guides/cn/tv.cctv.com.epg.xml`
        *   **External:** `http://epg.51zmt.top:8000/e.xml`

    ![jellyfin-epg](./image/jellyfin-epg.jpg)

## References

*   [BurningC4/Chinese-IPTV](https://github.com/BurningC4/Chinese-IPTV)
*   [SoPudge/kodi_iptv_epg](https://github.com/SoPudge/kodi_iptv_epg)
*   [BurningC4/getepg](https://github.com/BurningC4/getepg)
*   [3mile/cctv_api_get_EPG](https://github.com/3mile/cctv_api_get_EPG)
*   [国内高清直播live - TV001](http://www.tv001.vip/forum.php?mod=viewthread&tid=3)
*   [广东移动某河全套 - 恩山无线论坛](https://www.right.com.cn/forum/thread-6809023-1-1.html)

## Changelog

*   211126: Marked some sources as unusable, and added China Mobile source
*   211123: Fixed EPG update issues, and added another EPG source
*   211122: Added EPG guide `guide.xml` with daily auto-updates
*   211122: Split into general and Guangdong province-specific versions
*   211121: Initial release

**Thanks to the open internet! 🎉🎉🎉**