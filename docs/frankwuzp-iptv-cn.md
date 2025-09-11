# IPTV-CN: Free IPTV Resources for Jellyfin and More

**Looking for a reliable and up-to-date source for Chinese IPTV channels?** This repository provides easily accessible IPTV resources, specifically tailored for use with Jellyfin and other IPTV-compatible platforms. ([Original Repo](https://github.com/frankwuzp/iptv-cn))

[![GitHub last commit](https://img.shields.io/github/last-commit/frankwuzp/iptv-cn/main?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![GitHub repo size](https://img.shields.io/github/repo-size/frankwuzp/iptv-cn?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![jsdelivr](https://data.jsdelivr.com/v1/package/gh/frankwuzp/iptv-cn/badge)](https://www.jsdelivr.com/package/gh/frankwuzp/iptv-cn)
[![GitHub watchers](https://img.shields.io/github/watchers/frankwuzp/iptv-cn?style=social)](https://github.com/frankwuzp/iptv-cn)

## Key Features:

*   **Updated IPTV Channel Lists:** Provides frequently updated `.m3u` files containing working IPTV channel URLs.
*   **Optimized for Jellyfin:** Designed for seamless integration with Jellyfin's live TV features.
*   **Multiple Channel Sources:** Includes channel lists sourced from various providers, including China Mobile, for broader compatibility.
*   **Automated EPG (Electronic Program Guide):** Offers an automatically updated EPG file (`guide.xml`) for accurate program information.
*   **CDN Acceleration:** Uses jsDelivr CDN for faster access, particularly for users in mainland China.

## Available Files:

*   `tv-ipv4-cn.m3u`: General-purpose Chinese IPTV channel list.
*   `tv-ipv4-cmcc.m3u`: Channel list optimized for China Mobile users.
*   `guide.xml`: Electronic Program Guide, automatically updated daily.
*   `requirements.txt`: Python dependencies for the EPG generation script (`get-epg.py`).

## How to Use:

### 1. Get the Channel List (.m3u)

Choose one of the following methods to obtain the channel list:

*   **GitHub Raw:** `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cmcc.m3u`
*   **jsDelivr CDN (Recommended for China):** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cmcc.m3u`

You can also download the `tv-ipv4-cmcc.m3u` file directly from this repository.

### 2. Configure Jellyfin

1.  In your Jellyfin server, go to "Live TV" settings.
2.  Add a new IPTV provider.
3.  Enter the URL of the `.m3u` file (e.g., the jsDelivr CDN link above) into the "M3U Playlist URL" field.
4.  (Optional) Add the EPG file URL to provide program information.

### 3. Get the Guide File (EPG)

Choose one of the following guide file options:

*   **GitHub Raw:** `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/guide.xml`
*   **jsDelivr CDN (Recommended for China):** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/guide.xml`
*   **iptv-org:** `https://iptv-org.github.io/epg/guides/cn/tv.cctv.com.epg.xml`
*   **Alternative EPG Source:** `http://epg.51zmt.top:8000/e.xml` (use with caution).

Use the URL of your selected guide file in your Jellyfin setup.

## References:

*   [BurningC4/Chinese-IPTV](https://github.com/BurningC4/Chinese-IPTV)
*   [SoPudge/kodi_iptv_epg](https://github.com/SoPudge/kodi_iptv_epg)
*   [BurningC4/getepg](https://github.com/BurningC4/getepg)
*   [3mile/cctv_api_get_EPG](https://github.com/3mile/cctv_api_get_EPG)
*   [国内高清直播live - TV001](http://www.tv001.vip/forum.php?mod=viewthread&tid=3)
*   [广东移动某河全套 - 恩山无线论坛](https://www.right.com.cn/forum/thread-6809023-1-1.html)

## Changelog:

*   **2021-11-26:** Marked non-working channel sources; added a China Mobile source.
*   **2021-11-23:** Fixed issues with EPG updates and added an additional EPG source.
*   **2021-11-22:** Implemented automatic EPG updates (daily at 1 AM and 6 AM) and added `guide.xml`.
*   **2021-11-22:** Separated channel lists into general and Guangdong-specific versions.
*   **2021-11-21:** Initial release.