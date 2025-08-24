# Watch Chinese IPTV Channels on Jellyfin with IPTV-CN

Access and enjoy a curated list of working Chinese IPTV channels, perfect for streaming on Jellyfin.  **(Visit the original repo for the latest updates: [frankwuzp/iptv-cn](https://github.com/frankwuzp/iptv-cn))**

[![GitHub last commit (branch)](https://img.shields.io/github/last-commit/frankwuzp/iptv-cn/main?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![GitHub repo size](https://img.shields.io/github/repo-size/frankwuzp/iptv-cn?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![jsdelivr](https://data.jsdelivr.com/v1/package/gh/frankwuzp/iptv-cn/badge)](https://www.jsdelivr.com/package/gh/frankwuzp/iptv-cn)
[![GitHub watchers](https://img.shields.io/github/watchers/frankwuzp/iptv-cn?style=social)](https://github.com/frankwuzp/iptv-cn)

## Key Features

*   **Working IPTV Channels:** Includes a selection of tested Chinese IPTV channels, optimized for use with Jellyfin.
*   **Optimized for Guangdong (Guangdong) Province:** Provides resources that are tested and suitable for Guangdong users.
*   **Mobile Signal Sources:** Includes channels from China Mobile (CMCC) for broader compatibility.
*   **Easy Integration with Jellyfin:**  Provides clear instructions and links for configuring IPTV streams in Jellyfin.
*   **EPG (Electronic Program Guide) Support:** Includes a guide file (`guide.xml`) that is automatically updated daily, providing program information.
*   **CDN Acceleration:** Uses jsDelivr CDN to improve streaming performance for users in Mainland China.

## How to Use

### File Descriptions

*   `tv-ipv4-cn`: General-purpose `.m3u` file for Chinese IPTV channels.
*   `tv-ipv4-cmcc`: IPTV channels sourced from China Mobile (CMCC). (Recommended)
*   `tv-ipv4-old`: Archive of older IPTV resources, based on BurningC4's repository (may have limited availability).
*   `guide.xml`: Electronic Program Guide (EPG) file, automatically updated daily at 1 AM and 6 AM via GitHub Actions.
*   `requirements.txt`: Python dependencies for the `get-epg.py` script.

### Channel Lists (Example: Guangdong)

Use the following `.m3u` URL in your Jellyfin TV Live configuration:

*   **GitHub:**
    `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cmcc.m3u`

*   **jsDelivr CDN (Recommended for Mainland China):**
    `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cmcc.m3u`

**Steps:**

1.  Save the `tv-ipv4-cmcc.m3u` file (or use the direct URLs above).
2.  In your Jellyfin settings, add the URL to your IPTV provider.

    ![jellyfin-setting](./image/jellyfin-settings.jpg)

### Guide File (EPG) Sources

Use one of the following EPG URLs in your Jellyfin TV Live configuration:

*   **"Mysterious God" Source:**
    `http://epg.51zmt.top:8000/e.xml`
*   **GitHub:**
    `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/guide.xml`
*   **jsDelivr CDN (Recommended for Mainland China):**
    `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/guide.xml`
*   **iptv-org:**
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

*   211126:  Notes on unavailable streams; added China Mobile (CMCC) source.
*   211123:  Fixed EPG update issue; added an EPG source.
*   211122:  Added EPG (`guide.xml`) with automatic updates (daily at 1 AM and 6 AM).
*   211122:  Separated channels into general and Guangdong-specific versions.
*   211121:  Initial release.