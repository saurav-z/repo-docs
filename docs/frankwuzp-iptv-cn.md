# IPTV CN: Free Live TV Channels for Jellyfin

**Get access to free live TV channels, specifically tailored for Jellyfin, with this easy-to-use IPTV resource.  ([View on GitHub](https://github.com/frankwuzp/iptv-cn))**

[![GitHub last commit](https://img.shields.io/github/last-commit/frankwuzp/iptv-cn/main?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![GitHub repo size](https://img.shields.io/github/repo-size/frankwuzp/iptv-cn?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![jsdelivr badge](https://data.jsdelivr.com/v1/package/gh/frankwuzp/iptv-cn/badge)](https://www.jsdelivr.com/package/gh/frankwuzp/iptv-cn)
[![GitHub watchers](https://img.shields.io/github/watchers/frankwuzp/iptv-cn?style=social)](https://github.com/frankwuzp/iptv-cn)

## Key Features:

*   **Free IPTV Channels:** Access a curated list of free live TV channels.
*   **Jellyfin Compatible:** Optimized for seamless integration with Jellyfin's live TV feature.
*   **Updated Resources:**  Regularly updated channel lists and EPG (Electronic Program Guide) data.
*   **Multiple Channel Sources:** Includes sources for general use and mobile-specific streams.
*   **Automatic EPG Updates:** The guide.xml is auto-updated daily.
*   **CDN Support:** Uses jsDelivr CDN for faster access, especially for users in mainland China.

## Channel Lists & Usage

This repository provides `.m3u` files containing the channel lists, optimized for use with Jellyfin. You can use these files directly in your Jellyfin setup to stream live TV.

### Available Files:

*   `tv-ipv4-cn.m3u`: General use channel list.
*   `tv-ipv4-cmcc.m3u`:  Mobile signal source (tested and working).
*   `tv-ipv4-old.m3u`:  Older channel list (some channels may still work).

### How to Use with Jellyfin

1.  **Get the M3U URL:** Choose your preferred channel list file.  You can find the raw URLs in the "How to Use" section of the original README.
    *   **From GitHub:** `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cmcc.m3u`
    *   **From jsDelivr CDN (recommended for users in mainland China):** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cmcc.m3u`
2.  **Add to Jellyfin:**  In your Jellyfin server settings, add the URL of the `.m3u` file to the "Live TV" section.

    ![jellyfin-setting](./image/jellyfin-settings.jpg)

### Electronic Program Guide (EPG)

An EPG provides TV listings for each channel.  This repository provides an automatically updated EPG file.

**Available EPG Sources (choose one):**

*   **From Github:** `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/guide.xml`
*   **From jsDelivr CDN (optimized for mainland users):** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/guide.xml`
*   **Alternative EPG Sources** (See the original README):

![jellyfin-epg](./image/jellyfin-epg.jpg)

## Changelog:

*   **2021-11-26:** Updated with information about non-working sources and added the mobile source.
*   **2021-11-23:** Fixed EPG update issues and added a new EPG source.
*   **2021-11-22:** Added EPG `guide.xml` with automatic updates, and separated channel lists.
*   **2021-11-21:** Initial release.

## References:

*   See the original README for a complete list of references.