[![Discord](https://img.shields.io/discord/784619049208250388?style=plastic)][discord] [![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&style=plastic&up_color=blue&up_message=online&url=https%3A%2F%2Flibrephotos.com)](https://librephotos.com/)
[![Read the docs](https://img.shields.io/static/v1?label=Read&message=the%20docs&color=blue&style=plastic)](https://docs.librephotos.com/) [![GitHub contributors](https://img.shields.io/github/contributors/librephotos/librephotos?style=plastic)](https://github.com/LibrePhotos/librephotos/graphs/contributors)
<a href="https://hosted.weblate.org/engage/librephotos/">
<img src="https://hosted.weblate.org/widgets/librephotos/-/librephotos-frontend/svg-badge.svg" alt="Translation status" />
</a>

# LibrePhotos: Your Self-Hosted, Open-Source Photo Management Solution

Tired of relying on closed-source photo services?  LibrePhotos offers a powerful, open-source alternative for managing and organizing your entire photo and video collection, giving you complete control.  Check out the [LibrePhotos repository on GitHub](https://github.com/LibrePhotos/librephotos) for more details.

## Key Features

*   **Comprehensive Media Support:** Supports photos (including RAW files) and videos.
*   **Intuitive Organization:**  Offers a timeline view and automatic album generation based on events.
*   **Intelligent Search:** Features face recognition, object/scene detection, semantic image search, and metadata-based searching.
*   **Multi-User Capabilities:** Allows multiple users to access and manage photos.
*   **Automated Photo Import:** Scans your file system to automatically import photos.
*   **Built-in Reverse Geocoding:** Adds location data to your photos for easy browsing.

## Getting Started

*   **Demo:** Experience LibrePhotos firsthand via the [stable demo](https://demo1.librephotos.com/) (user: `demo`, password: `demo1234`) or the [development demo](https://demo2.librephotos.com/) (same credentials).
*   **Installation:**  Follow the detailed step-by-step instructions in our [documentation](https://docs.librephotos.com/docs/installation/standard-install).
*   **Development Videos:** Watch development progress on [Niaz Faridani-Rad's YouTube channel](https://www.youtube.com/channel/UCZJ2pk2BPKxwbuCV9LWDR0w).
*   **Community:** Join the LibrePhotos community on [Discord][discord] to connect with other users and developers.

## How to Contribute

We welcome contributions from the community!

*   ⭐ **Star** the repository on GitHub to show your support.
*   🚀 **Develop:** Get started developing in under 30 minutes by following [this guide](https://docs.librephotos.com/docs/development/dev-install).
*   🗒️ **Document:**  Improve the documentation by submitting pull requests [here](https://github.com/LibrePhotos/librephotos.docs).
*   🧪 **Test:** Help find bugs by using the `dev` tag and reporting issues.
*   🧑‍🤝‍🧑 **Outreach:**  Spread the word about LibrePhotos.
*   🌐 **Translate:**  Make LibrePhotos accessible to more people through [Weblate](https://hosted.weblate.org/engage/librephotos/).
*   💸 **Donate:**  Support the developers via [GitHub Sponsors](https://github.com/sponsors/derneuere).

## Technologies Used

LibrePhotos leverages several powerful open-source tools:

*   **Image Conversion:** [ImageMagick](https://github.com/ImageMagick/ImageMagick)
*   **Video Conversion:** [FFmpeg](https://github.com/FFmpeg/FFmpeg)
*   **Exif Support:** [ExifTool](https://github.com/exiftool/exiftool)
*   **Face Detection:** [face_recognition](https://github.com/ageitgey/face_recognition)
*   **Face Classification/Clustering:** [scikit-learn](https://scikit-learn.org/) and [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
*   **Image Captioning:** [im2txt](https://github.com/HughKu/Im2txt)
*   **Scene Classification:** [places365](http://places.csail.mit.edu/)
*   **Reverse Geocoding:** [Mapbox](https://www.mapbox.com/) (requires an API key; first 50,000 geocode lookups free per month)

[discord]: https://discord.gg/xwRvtSDGWb