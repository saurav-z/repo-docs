# IPTV-CN: Watch Chinese TV Channels with Jellyfin (Updated & Optimized)

**Tired of missing your favorite Chinese TV shows?** IPTV-CN provides up-to-date IPTV resources, specifically optimized for use with Jellyfin, enabling you to easily stream Chinese TV channels. ([View the Original Repo](https://github.com/frankwuzp/iptv-cn))

[![GitHub last commit (branch)](https://img.shields.io/github/last-commit/frankwuzp/iptv-cn/main?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![GitHub repo size](https://img.shields.io/github/repo-size/frankwuzp/iptv-cn?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![jsdelivr](https://data.jsdelivr.com/v1/package/gh/frankwuzp/iptv-cn/badge)](https://www.jsdelivr.com/package/gh/frankwuzp/iptv-cn)
[![GitHub watchers](https://img.shields.io/github/watchers/frankwuzp/iptv-cn?style=social)](https://github.com/frankwuzp/iptv-cn)


## Key Features:

*   **Reliable IPTV Resources:** Access tested and working IPTV streams for Chinese TV channels.
*   **Jellyfin Compatibility:** Optimized for seamless integration with your Jellyfin media server.
*   **Automatic EPG Updates:**  Enjoy regularly updated Electronic Program Guides (EPGs) for accurate scheduling.
*   **Multiple Stream Sources:** Choose from different stream sources, including ones optimized for mainland China users, and mobile providers.
*   **Easy Setup:** Simple instructions for integrating channels and EPGs into your Jellyfin setup.

## Available IPTV Streams:

The following `.m3u` files provide the channel listings:

*   `tv-ipv4-cn`: General-purpose IPTV list for channels across China.
*   `tv-ipv4-cmcc`:  Updated Mobile Channels (tested and working).
*   `tv-ipv4-old`: Legacy channels from the original source ([BurningC4/Chinese-IPTV](https://github.com/BurningC4/Chinese-IPTV)), may experience issues.

### Accessing Channel Lists:

You can access the channel lists via the following methods:

*   **GitHub Raw:**
    `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cmcc.m3u` (Example, replace with your desired file name)

*   **jsDelivr CDN (Recommended for Mainland China Users - Faster CDN):**
    `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cmcc.m3u` (Example, replace with your desired file name)

**To use these with Jellyfin:**

1.  Save the desired `.m3u` file from the provided links or copy the URL.
2.  In your Jellyfin settings, navigate to "Live TV" -> "TV Channels" and add a new channel source.
3.  Paste the URL of the `.m3u` file into the "URL" field.
    ![jellyfin-setting](./image/jellyfin-settings.jpg)

## Electronic Program Guide (EPG)

EPG data provides program schedules for each channel, helping you plan your viewing.

### Available EPG Sources (Choose one):

*   **Mysterious Source:**
    `http://epg.51zmt.top:8000/e.xml`

*   **GitHub Raw:**
    `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/guide.xml`

*   **jsDelivr CDN (Optimized for Mainland Users):**
    `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/guide.xml`

*   **iptv-org:**
    `https://iptv-org.github.io/epg/guides/cn/tv.cctv.com.epg.xml`

**To use with Jellyfin:**

1.  Copy the URL of your chosen EPG source.
2.  In your Jellyfin settings, under "Live TV" -> "Guide Providers", add a new guide provider.
3.  Paste the URL into the "URL" field.
    ![jellyfin-epg](./image/jellyfin-epg.jpg)

## References:

*   [BurningC4/Chinese-IPTV](https://github.com/BurningC4/Chinese-IPTV)
*   [SoPudge/kodi_iptv_epg](https://github.com/SoPudge/kodi_iptv_epg)
*   [BurningC4/getepg](https://github.com/BurningC4/getepg)
*   [3mile/cctv_api_get_EPG](https://github.com/3mile/cctv_api_get_EPG)
*   [国内高清直播live - TV001](http://www.tv001.vip/forum.php?mod=viewthread&tid=3)
*   [广东移动某河全套 - 恩山无线论坛](https://www.right.com.cn/forum/thread-6809023-1-1.html)

**Thank you to the open-source community for making this possible! 🎉**

## Changelog

*   **211126:** Marked unusable stream sources; added working mobile signal sources.
*   **211123:** Fixed EPG update issues and added an additional EPG source.
*   **211122:** Introduced `guide.xml` for EPG (updated daily at 1 AM and 6 AM).
*   **211122:** Separated channel lists by region (general and Guangdong).
*   **211121:** Initial release.