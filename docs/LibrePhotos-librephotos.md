[![Discord](https://img.shields.io/discord/784619049208250388?style=plastic)][discord]
[![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&style=plastic&up_color=blue&up_message=online&url=https%3A%2F%2Flibrephotos.com)](https://librephotos.com/)
[![Read the docs](https://img.shields.io/static/v1?label=Read&message=the%20docs&color=blue&style=plastic)](https://docs.librephotos.com/)
[![GitHub contributors](https://img.shields.io/github/contributors/librephotos/librephotos?style=plastic)](https://github.com/LibrePhotos/librephotos/graphs/contributors)
<a href="https://hosted.weblate.org/engage/librephotos/">
<img src="https://hosted.weblate.org/widgets/librephotos/-/librephotos-frontend/svg-badge.svg" alt="Translation status" />
</a>

# LibrePhotos: Your Self-Hosted, Open-Source Photo Management Solution

Effortlessly organize and enjoy your photos with LibrePhotos, a powerful and privacy-focused alternative to proprietary photo services.

[Visit the original repository on GitHub](https://github.com/LibrePhotos/librephotos)

![LibrePhotos Screenshot](https://github.com/LibrePhotos/librephotos/blob/dev/screenshots/mockups_main_fhd.png?raw=true)
<sub>Mockup designed by rawpixel.com / Freepik</sub>

## Key Features of LibrePhotos:

*   **Comprehensive Media Support:** Works with photos (including RAW files) and videos.
*   **Intelligent Organization:** Features a timeline view and automated album creation based on events.
*   **Advanced Search:** Semantic image search, and search by metadata.
*   **Facial Recognition & Classification:** Automatically identify and organize people in your photos.
*   **Object & Scene Detection:** Detects objects and scenes within your images for smarter searching and organization.
*   **Multi-User Support:** Share your photos with others while maintaining privacy.
*   **Geolocation Capabilities:** Utilizes reverse geocoding to map your photos' locations.
*   **Seamless File System Integration:** Efficiently scans your file system for photos and videos.

## Getting Started

*   **Live Demos:**
    *   **Stable:** [https://demo1.librephotos.com/](https://demo1.librephotos.com/) (user: `demo`, password: `demo1234`)
    *   **Development:** [https://demo2.librephotos.com/](https://demo2.librephotos.com/) (user: `demo`, password: `demo1234`)
*   **Installation:** Detailed instructions are available in the [LibrePhotos documentation](https://docs.librephotos.com/docs/installation/standard-install).
*   **Development Videos:** Watch development videos on [Niaz Faridani-Rad's YouTube channel](https://www.youtube.com/channel/UCZJ2pk2BPKxwbuCV9LWDR0w).
*   **Join the Community:** Connect with other users and developers on our [Discord][discord].

## How to Contribute

We welcome contributions from everyone! Here's how you can help:

*   ⭐ **Star the repository** to show your support.
*   🚀 **Develop:**  Follow [this guide](https://docs.librephotos.com/docs/development/dev-install) to start developing in under 30 minutes.
*   🗒️ **Document:** Help improve the documentation by submitting pull requests [here](https://github.com/LibrePhotos/librephotos.docs).
*   🧪 **Test:**  Use the `dev` tag and report any bugs you find by opening an issue.
*   🧑‍🤝‍🧑 **Spread the Word:** Tell others about LibrePhotos and encourage them to try it out.
*   🌐 **Translate:**  Help make LibrePhotos accessible to more people through translations on [weblate](https://hosted.weblate.org/engage/librephotos/).
*   💸 **Donate:** Support the developers of LibrePhotos by donating through [GitHub Sponsors](https://github.com/sponsors/derneuere).

## Technologies Used

LibrePhotos leverages the following technologies:

*   **Image Conversion:** [ImageMagick](https://github.com/ImageMagick/ImageMagick)
*   **Video Conversion:** [FFmpeg](https://github.com/FFmpeg/FFmpeg)
*   **Exif Support:** [ExifTool](https://github.com/exiftool/exiftool)
*   **Face Detection:** [face\_recognition](https://github.com/ageitgey/face_recognition)
*   **Face Classification/Clusterization:** [scikit-learn](https://scikit-learn.org/) and [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
*   **Image Captioning:** [im2txt](https://github.com/HughKu/Im2txt)
*   **Scene Classification:** [places365](http://places.csail.mit.edu/)
*   **Reverse Geocoding:** [Mapbox](https://www.mapbox.com/) (requires an API key - the first 50,000 geocode lookups are free per month)

[discord]: https://discord.gg/xwRvtSDGWb