# IPTV-CN: Stream Chinese TV Channels with Jellyfin

**Access and stream Chinese TV channels directly in Jellyfin with this easy-to-use IPTV resource.**

[![GitHub last commit (branch)](https://img.shields.io/github/last-commit/frankwuzp/iptv-cn/main?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![GitHub repo size](https://img.shields.io/github/repo-size/frankwuzp/iptv-cn?style=flat-square)](https://github.com/frankwuzp/iptv-cn)
[![jsdelivr](https://data.jsdelivr.com/v1/package/gh/frankwuzp/iptv-cn/badge)](https://www.jsdelivr.com/package/gh/frankwuzp/iptv-cn)
[![GitHub watchers](https://img.shields.io/github/watchers/frankwuzp/iptv-cn?style=social)](https://github.com/frankwuzp/iptv-cn)

## Key Features

*   **Reliable IPTV Resources:** Provides working IPTV channel lists for Chinese TV, specifically tested for Guangdong (Guangdong) and now including a CMCC (China Mobile) source.
*   **Jellyfin Compatibility:** Designed for seamless integration with Jellyfin for live TV streaming.
*   **Up-to-Date Channel Listings:**  Includes regularly updated `.m3u` files for channel access.
*   **Automated EPG (Electronic Program Guide):**  Automatically generated and updated EPG (`guide.xml`) for program information.
*   **CDN Optimized:**  Offers jsDelivr CDN links for faster and more reliable access, especially for users in mainland China.
*   **Easy Setup:** Simple instructions and clear file descriptions for easy configuration within Jellyfin.
*   **Multiple EPG Options:**  Provides multiple EPG source options for flexibility.

## Getting Started

### Files Explained

*   `tv-ipv4-cn.m3u`:  General channel list for mainland China.
*   `tv-ipv4-cmcc.m3u`:  Channel list specifically for China Mobile users (tested and working).
*   `tv-ipv4-old.m3u`: Archive of older channels for reference (some may still work, but might have delays or buffering).
*   `guide.xml`:  Electronic Program Guide (EPG) automatically updated daily at 1 AM and 6 AM via GitHub Actions.
*   `requirements.txt`: Dependencies for the Python script (`get-epg.py`) used to generate the EPG.

### How to Use in Jellyfin

1.  **Choose a Channel List:** Select either:
    *   `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cmcc.m3u` (GitHub)
    *   `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cmcc.m3u` (jsDelivr CDN - recommended for users in mainland China)
    *   Or download the `tv-ipv4-cmcc.m3u` file and use it.
2.  **Add the Channel List to Jellyfin:** Go to your Jellyfin server settings, and add the URL (from step 1) as your IPTV provider URL.  See the example image in the original README (shown as `jellyfin-settings.jpg`) for the correct location to input the URL.

### Guide File (EPG) Options

Choose ONE of the following EPG sources:

*   `http://epg.51zmt.top:8000/e.xml` (Mysterious Source)
*   `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/guide.xml` (GitHub)
*   `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/guide.xml` (jsDelivr CDN - optimized for mainland users)
*   `https://iptv-org.github.io/epg/guides/cn/tv.cctv.com.epg.xml` (iptv-org)

See the example image in the original README (shown as `jellyfin-epg.jpg`) for the correct location to input the EPG URL.

## Resources and References

*   [Original Repository on GitHub](https://github.com/frankwuzp/iptv-cn)
*   [BurningC4/Chinese-IPTV](https://github.com/BurningC4/Chinese-IPTV)
*   [SoPudge/kodi_iptv_epg](https://github.com/SoPudge/kodi_iptv_epg)
*   [BurningC4/getepg](https://github.com/BurningC4/getepg)
*   [3mile/cctv_api_get_EPG](https://github.com/3mile/cctv_api_get_EPG)
*   [国内高清直播live - TV001](http://www.tv001.vip/forum.php?mod=viewthread&tid=3)
*   [广东移动某河全套 - 恩山无线论坛](https://www.right.com.cn/forum/thread-6809023-1-1.html)

**Thanks to the open internet!**

## Changelog

*   **2021-11-26:**  Added China Mobile channel source and updated channel source status
*   **2021-11-23:**  Fixed EPG update issues and added a new EPG source.
*   **2021-11-22:**  Implemented automated EPG generation and added a new EPG file.
*   **2021-11-22:**  Separated channel lists into general and Guangdong-specific versions.
*   **2021-11-21:**  Initial Release
```

Key improvements in this version:

*   **SEO Optimization:**  Includes relevant keywords like "IPTV," "Chinese TV," "Jellyfin," "China," and "Guangdong" in the title and description.
*   **Clear Headings and Structure:** Uses headings and bullet points to make the README easy to read and understand.
*   **Concise Summary:** Starts with a one-sentence hook to grab the reader's attention.
*   **Key Features Section:** Highlights the main benefits of using the repository.
*   **Clear Instructions:** Simplifies the instructions for adding the channel list and EPG to Jellyfin.
*   **CDN Recommendation:**  Emphasizes the use of the CDN for better performance in China.
*   **Complete Information:**  Includes all the necessary links and information from the original README.
*   **Changelog Retention:** Keeps the changelog for historical information.
*   **Hyperlinks to Original Repo:** Adds link back to the original repo for easy navigation.