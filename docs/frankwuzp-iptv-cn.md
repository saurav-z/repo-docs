# IPTV-CN: Watch Chinese TV Channels with Jellyfin 🇨🇳

Easily stream live Chinese TV channels in Jellyfin using this repository, providing up-to-date IPTV resources.  [View the original repository](https://github.com/frankwuzp/iptv-cn).

[![GitHub last commit](https://img.shields.io/github/last-commit/frankwuzp/iptv-cn/main?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![GitHub repo size](https://img.shields.io/github/repo-size/frankwuzp/iptv-cn?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![jsdelivr](https://data.jsdelivr.com/v1/package/gh/frankwuzp/iptv-cn/badge)](https://www.jsdelivr.com/package/gh/frankwuzp/iptv-cn)
[![GitHub watchers](https://img.shields.io/github/watchers/frankwuzp/iptv-cn?style=social)](https://github.com/frankwuzp/iptv-cn)

## Key Features

*   **Live Chinese TV:** Access a curated list of Chinese TV channels for streaming.
*   **Jellyfin Compatibility:** Designed for easy integration with Jellyfin for live TV.
*   **Optimized for Mainland China:** Includes CDN options for faster loading in China.
*   **Automatic EPG Updates:**  `guide.xml` is automatically updated daily to provide up-to-date TV listings (EPG).
*   **Multiple Source Options:** Choose from various sources for channel lists and EPG data.
*   **Includes Mobile Sources:** Includes mobile sources (cmcc) for channels, where applicable.

## Getting Started

### IPTV Channel Lists (M3U Files)

Choose an M3U file and input it into your Jellyfin TV Live settings.  Here are your options:

*   `tv-ipv4-cn.m3u`: General Chinese Channels
*   `tv-ipv4-cmcc.m3u`:  Channels via China Mobile, tested and working.

#### Accessing the Channel Lists

You can access the files via:

*   **Github (raw):**  `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cmcc.m3u` (replace with the m3u you want)
*   **jsDelivr CDN (Recommended for Mainland China):** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cmcc.m3u` (replace with the m3u you want)

1.  Copy the appropriate URL from above.
2.  In your Jellyfin server, go to **Live TV** settings.
3.  Add a new tuner and paste the URL into the **Network URL** field.
4.  Save the tuner.

### Electronic Program Guide (EPG)

Choose an EPG source to get program listings.

*   `http://epg.51zmt.top:8000/e.xml` (External Source)
*   **Github (raw):**  `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/guide.xml`
*   **jsDelivr CDN (Optimized for Mainland China):** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/guide.xml`
*   **iptv-org:** `https://iptv-org.github.io/epg/guides/cn/tv.cctv.com.epg.xml`

1.  Copy the appropriate URL from above.
2.  In your Jellyfin server, go to **Live TV** settings.
3.  Configure the EPG provider within your Jellyfin settings.
4.  Paste the URL of your chosen EPG source.

## References

*   [BurningC4/Chinese-IPTV](https://github.com/BurningC4/Chinese-IPTV)
*   [SoPudge/kodi_iptv_epg](https://github.com/SoPudge/kodi_iptv_epg)
*   [BurningC4/getepg](https://github.com/BurningC4/getepg)
*   [3mile/cctv_api_get_EPG](https://github.com/3mile/cctv_api_get_EPG)
*   [国内高清直播live - TV001](http://www.tv001.vip/forum.php?mod=viewthread&tid=3)
*   [广东移动某河全套 - 恩山无线论坛](https://www.right.com.cn/forum/thread-6809023-1-1.html)

## Changelog

*   **2021-11-26:** Added mobile signal sources and updated availability notes.
*   **2021-11-23:** Improved EPG update process and added a new EPG source.
*   **2021-11-22:** Introduced automatic EPG updates (daily at 1 AM and 6 AM) and added `guide.xml`.
*   **2021-11-22:** Separated into general and Guangdong-specific versions.
*   **2021-11-21:** Initial release.