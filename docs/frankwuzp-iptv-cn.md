# IPTV CN: Stream Chinese TV Channels with Jellyfin

**Easily access and stream live Chinese TV channels directly within your Jellyfin media server.** ([View on GitHub](https://github.com/frankwuzp/iptv-cn))

This repository provides up-to-date IPTV resources, specifically curated for Jellyfin users, focusing on reliable Chinese TV streams.

## Key Features:

*   **Reliable IPTV Resources:** Provides `.m3u` playlists with working IPTV channels, tested and optimized for Jellyfin.
*   **Optimized for Mainland China:** Includes CDN links for faster streaming and improved performance within China.
*   **Automatic EPG Updates:**  The `guide.xml` file is automatically updated daily at 1 AM and 6 AM to ensure accurate TV listings.
*   **Multiple Channel Source Options:** Includes channels from both generic and CMCC sources, offering diverse viewing choices.
*   **Easy Integration with Jellyfin:** Provides clear instructions and links for integrating the IPTV resources into your Jellyfin setup.

## Resources and Usage:

### M3U Playlist Files:

These files contain the channel lists that you can use in your Jellyfin setup.

*   `tv-ipv4-cn`:  Generic Chinese channels.
*   `tv-ipv4-cmcc`: Chinese channels from CMCC source. (Recommended for improved reliability).
*   `tv-ipv4-old`: (Legacy) Contains older channel resources (some may be outdated).

#### Accessing the M3U Files:

You can use the following URLs in your Jellyfin TV Live settings (choose one):

*   **GitHub:**  `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/tv-ipv4-cmcc.m3u`
*   **jsDelivr CDN (Recommended for Mainland China):** `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/tv-ipv4-cmcc.m3u`

### Electronic Program Guide (EPG) Files:

EPG files provide TV listings information for your channels.

*   `guide.xml`: Automatically updated daily, provides program information.

#### Accessing the EPG File:

You can use the following URLs in your Jellyfin TV Live settings (choose one):

*   **GitHub:**  `https://raw.githubusercontent.com/frankwuzp/iptv-cn/main/guide.xml`
*   **jsDelivr CDN (Recommended for Mainland China):**  `https://cdn.jsdelivr.net/gh/frankwuzp/iptv-cn@latest/guide.xml`
*   **External Source:** `http://epg.51zmt.top:8000/e.xml`
*   **iptv-org:** `https://iptv-org.github.io/epg/guides/cn/tv.cctv.com.epg.xml`

### Integrating with Jellyfin:

1.  Go to your Jellyfin server settings.
2.  Navigate to "Live TV" -> "Manage Providers".
3.  Add a new provider using one of the M3U playlist URLs provided above.
4.  Add a new EPG source using one of the XML file URLs provided above.
5.  Save your settings.

## References:

*   [BurningC4/Chinese-IPTV](https://github.com/BurningC4/Chinese-IPTV)
*   [SoPudge/kodi_iptv_epg](https://github.com/SoPudge/kodi_iptv_epg)
*   [BurningC4/getepg](https://github.com/BurningC4/getepg)
*   [3mile/cctv_api_get_EPG](https://github.com/3mile/cctv_api_get_EPG)
*   [国内高清直播live - TV001](http://www.tv001.vip/forum.php?mod=viewthread&tid=3)
*   [广东移动某河全套 - 恩山无线论坛](https://www.right.com.cn/forum/thread-6809023-1-1.html)

## Changelog:

*   **211126:** Updated channel sources and added CMCC channels; marked some outdated sources.
*   **211123:** Resolved EPG update issues; added a new EPG source.
*   **211122:** Implemented automatic EPG updates; introduced `guide.xml`
*   **211122:** Separated channel lists into generic and province-specific versions.
*   **211121:** Initial commit.