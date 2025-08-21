# IPTV-CN: Free IPTV Resources for Jellyfin (China)

**Access reliable Chinese IPTV streams and TV guides for your Jellyfin setup with this easy-to-use resource.**  [View the original repository on GitHub](https://github.com/frankwuzp/iptv-cn).

[![Last Commit](https://img.shields.io/github/last-commit/frankwuzp/iptv-cn/main?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![Repo Size](https://img.shields.io/github/repo-size/frankwuzp/iptv-cn?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![jsdelivr](https://data.jsdelivr.com/v1/package/gh/frankwuzp/iptv-cn/badge)](https://www.jsdelivr.com/package/gh/frankwuzp/iptv-cn)
[![GitHub Watchers](https://img.shields.io/github/watchers/frankwuzp/iptv-cn?style=social)](https://github.com/frankwuzp/iptv-cn)

## Key Features:

*   **Reliable IPTV Streams:** Access working IPTV resources, specifically tested for Guangdong (China).
*   **Jellyfin Compatibility:** Designed for easy integration with Jellyfin's live TV functionality.
*   **Multiple Stream Sources:** Includes both general and China Mobile (CMCC) stream sources.
*   **Automated EPG Updates:** Get up-to-date Electronic Program Guide (EPG) data with daily updates.
*   **CDN Support:**  Utilizes jsDelivr CDN for faster access and improved performance for users in mainland China.

## Available Files:

*   `tv-ipv4-cn.m3u`: General IPTV stream file (China).
*   `tv-ipv4-cmcc.m3u`:  China Mobile (CMCC) stream file (tested and working).
*   `tv-ipv4-old.m3u`: Older stream file (may have some issues).
*   `guide.xml`: Electronic Program Guide (EPG) file, automatically updated daily at 1 AM and 6 AM (UTC).

## How to Use:

### 1. Channel List (M3U)

Choose *one* of the following URLs for your M3U file:

*   **GitHub (Raw):** `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cmcc.m3u`
*   **jsDelivr CDN (Recommended for Mainland China):** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cmcc.m3u`

  Save the `tv-ipv4-cmcc.m3u` file from this repository or use the above URLs in your Jellyfin Live TV settings.

  ![Jellyfin Settings Example](./image/jellyfin-settings.jpg)

### 2. Guide File (EPG - Electronic Program Guide)

Choose *one* of the following URLs for your EPG:

*   **GitHub (Raw):** `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/guide.xml`
*   **jsDelivr CDN (Optimized for Mainland China):** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/guide.xml`
*   **iptv-org:**  `https://iptv-org.github.io/epg/guides/cn/tv.cctv.com.epg.xml`
*   **Alternative EPG Source:** `http://epg.51zmt.top:8000/e.xml` (Use with caution)

![Jellyfin EPG Settings Example](./image/jellyfin-epg.jpg)

## Changelog:

*   **211126:** Added China Mobile (CMCC) streams and updated notes on stream availability.
*   **211123:** Fixed EPG update issues and added an alternative EPG source.
*   **211122:** Implemented automatic EPG updates and introduced the `guide.xml` file.
*   **211122:** Separated streams into general and Guangdong-specific versions.
*   **211121:** Initial release.

**Thank you to the open-source community for providing these resources!**