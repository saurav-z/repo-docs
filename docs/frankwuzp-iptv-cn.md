# IPTV-CN: Stream Chinese TV Channels with Ease

**Access and enjoy live Chinese TV channels directly within Jellyfin with this convenient IPTV resource.**  ([View the original repository](https://github.com/frankwuzp/iptv-cn))

[![GitHub last commit (branch)](https://img.shields.io/github/last-commit/frankwuzp/iptv-cn/main?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![GitHub repo size](https://img.shields.io/github/repo-size/frankwuzp/iptv-cn?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![jsdelivr](https://data.jsdelivr.com/v1/package/gh/frankwuzp/iptv-cn/badge)](https://www.jsdelivr.com/package/gh/frankwuzp/iptv-cn)
[![GitHub watchers](https://img.shields.io/github/watchers/frankwuzp/iptv-cn?style=social)](https://github.com/frankwuzp/iptv-cn)

## Key Features

*   **Reliable IPTV Resources:** Provides tested and working IPTV resources for Chinese TV channels, specifically optimized for Jellyfin.
*   **Multiple Channel Lists:** Offers different channel list options, including a general-purpose list (`tv-ipv4-cn`) and a list optimized for China Mobile users (`tv-ipv4-cmcc`).
*   **Easy Integration with Jellyfin:**  Simple instructions and URLs for seamless integration with your Jellyfin setup for live TV streaming.
*   **Automated EPG Updates:** Includes an Electronic Program Guide (EPG) that is automatically updated daily at 1 AM and 6 AM, ensuring up-to-date program information.
*   **CDN Support:** Provides jsDelivr CDN links for faster access and improved performance, especially for users in mainland China.

##  Getting Started

### Available Files

*   `tv-ipv4-cn.m3u`:  General-purpose IPTV playlist for Chinese channels.
*   `tv-ipv4-cmcc.m3u`:  IPTV playlist optimized for China Mobile users. (Recommended)
*   `tv-ipv4-old.m3u`:  Older channel list (may have compatibility issues).
*   `guide.xml`:  Electronic Program Guide (EPG) file, automatically updated.
*   `requirements.txt`: Python package dependencies for the `get-epg.py` script.

###  How to Use (Example with Jellyfin)

1.  **Choose a Channel List:** Select either:
    *   **GitHub:** `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cmcc.m3u`
    *   **jsDelivr CDN (Recommended for Mainland China):** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cmcc.m3u`
    
2.  **Configure Jellyfin:**
    *   Go to your Jellyfin server settings and find the Live TV section.
    *   Add a new channel source and paste the URL of your selected channel list.  (See the provided image [jellyfin-setting](./image/jellyfin-settings.jpg) in the original README.)

3.  **Choose an EPG Source:** Select an EPG source:
    *   **GitHub:** `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/guide.xml`
    *   **jsDelivr CDN (Recommended for Mainland China):** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/guide.xml`
    *   **iptv-org:** `https://iptv-org.github.io/epg/guides/cn/tv.cctv.com.epg.xml`
    *   **Alternative (Not Recommended):** `http://epg.51zmt.top:8000/e.xml`
    
4.  **Configure EPG in Jellyfin:**
    *   Enter the URL of your chosen EPG file in your Jellyfin Live TV settings. (See the provided image [jellyfin-epg](./image/jellyfin-epg.jpg) in the original README.)

##  References

*   [BurningC4/Chinese-IPTV](https://github.com/BurningC4/Chinese-IPTV)
*   [SoPudge/kodi_iptv_epg](https://github.com/SoPudge/kodi_iptv_epg)
*   [BurningC4/getepg](https://github.com/BurningC4/getepg)
*   [3mile/cctv_api_get_EPG](https://github.com/3mile/cctv_api_get_EPG)
*   [国内高清直播live - TV001](http://www.tv001.vip/forum.php?mod=viewthread&tid=3)
*   [广东移动某河全套 - 恩山无线论坛](https://www.right.com.cn/forum/thread-6809023-1-1.html)

## Changelog

*   211126:  Added note about non-working sources; added China Mobile source.
*   211123:  Fixed EPG update issues; added a new EPG source.
*   211122:  Added `guide.xml` for EPG with automatic updates (1 AM and 6 AM).
*   211122:  Separated lists for general and Guangdong province use.
*   211121:  Initial release.