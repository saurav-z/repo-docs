# IPTV China: Free Live TV Channels for Jellyfin

Looking for free, reliable live TV channels in China for your Jellyfin setup? This repository provides up-to-date M3U playlists and EPG (Electronic Program Guide) data for Chinese IPTV channels. **View the original repo on GitHub: [frankwuzp/iptv-cn](https://github.com/frankwuzp/iptv-cn)**

[![GitHub last commit (branch)](https://img.shields.io/github/last-commit/frankwuzp/iptv-cn/main?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![GitHub repo size](https://img.shields.io/github/repo-size/frankwuzp/iptv-cn?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![jsdelivr](https://data.jsdelivr.com/v1/package/gh/frankwuzp/iptv-cn/badge)](https://www.jsdelivr.com/package/gh/frankwuzp/iptv-cn)
[![GitHub watchers](https://img.shields.io/github/watchers/frankwuzp/iptv-cn?style=social)](https://github.com/frankwuzp/iptv-cn)

## Key Features

*   **Up-to-Date M3U Playlists:** Provides M3U files for live TV channels, including a China-wide general list (`tv-ipv4-cn.m3u`) and China Mobile (`tv-ipv4-cmcc.m3u`) channels.
*   **Electronic Program Guide (EPG):** Includes a `guide.xml` file for program listings, automatically updated daily via GitHub Actions.
*   **Jellyfin Compatibility:** Specifically designed for use with Jellyfin's live TV feature, enabling a seamless streaming experience.
*   **CDN Support:** Offers jsDelivr CDN links for faster download speeds, particularly beneficial for users in mainland China.
*   **Regular Updates:**  The repository is actively maintained, with playlist and EPG data updated frequently to ensure channel availability.

## How to Use

### Files Overview

*   `tv-ipv4-cn.m3u`: General M3U playlist for IPTV channels in China.
*   `tv-ipv4-cmcc.m3u`:  M3U playlist for China Mobile IPTV channels.
*   `tv-ipv4-old.m3u`:  Older playlist (from BurningC4/Chinese-IPTV) - might have some available channels, but with potential for lag and buffering.
*   `guide.xml`: Electronic Program Guide (EPG) data, automatically updated at 1 AM and 6 AM UTC via GitHub Actions using a Python script and the `requirements.txt` dependencies.

### Setting Up in Jellyfin

1.  **Get the M3U playlist URL:**
    *   **From GitHub:**
        `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cmcc.m3u` (Recommended for China Mobile)
        or, for general channels: `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cn.m3u`
    *   **From jsDelivr CDN (Faster for mainland China):**
        `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cmcc.m3u` (Recommended for China Mobile)
        or, for general channels: `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cn.m3u`

2.  **Add the playlist to Jellyfin:** In your Jellyfin server settings, go to "Live TV" and add the M3U URL to your TV tuner.

    ![jellyfin-setting](./image/jellyfin-settings.jpg)

3.  **Get the EPG file URL:**
    *   **From a Third-Party Source:**
        `http://epg.51zmt.top:8000/e.xml`
    *   **From GitHub:**
        `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/guide.xml`
    *   **From jsDelivr CDN (Faster for mainland China):**
        `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/guide.xml`
    *   **From iptv-org**
        `https://iptv-org.github.io/epg/guides/cn/tv.cctv.com.epg.xml`

4.  **Add the EPG to Jellyfin:** In your Jellyfin server settings, configure the EPG using one of the provided URLs.

    ![jellyfin-epg](./image/jellyfin-epg.jpg)

## References

*   [BurningC4/Chinese-IPTV](https://github.com/BurningC4/Chinese-IPTV)
*   [SoPudge/kodi_iptv_epg](https://github.com/SoPudge/kodi_iptv_epg)
*   [BurningC4/getepg](https://github.com/BurningC4/getepg)
*   [3mile/cctv_api_get_EPG](https://github.com/3mile/cctv_api_get_EPG)
*   [国内高清直播live - TV001](http://www.tv001.vip/forum.php?mod=viewthread&tid=3)
*   [广东移动某河全套 - 恩山无线论坛](https://www.right.com.cn/forum/thread-6809023-1-1.html)

## Changelog

*   **2021-11-26:** Marked unavailable streams, and added China Mobile streams.
*   **2021-11-23:** Fixed EPG update issues, and added a new EPG source.
*   **2021-11-22:** Added `guide.xml` EPG file and automated updates (1 AM & 6 AM UTC).
*   **2021-11-22:**  Split playlist into general and Guangdong-specific versions.
*   **2021-11-21:** Initial commit.