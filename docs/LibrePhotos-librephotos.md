# LibrePhotos: Your Open-Source, Self-Hosted Photo Management Solution

**Manage and organize your photo and video collection with ease using LibrePhotos, a powerful and open-source photo management solution.**

[![Discord](https://img.shields.io/discord/784619049208250388?style=plastic)][discord] [![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&style=plastic&up_color=blue&up_message=online&url=https%3A%2F%2Flibrephotos.com)](https://librephotos.com/)
[![Read the docs](https://img.shields.io/static/v1?label=Read&message=the%20docs&color=blue&style=plastic)](https://docs.librephotos.com/) [![GitHub contributors](https://img.shields.io/github/contributors/librephotos/librephotos?style=plastic)](https://github.com/LibrePhotos/librephotos/graphs/contributors)
<a href="https://hosted.weblate.org/engage/librephotos/">
<img src="https://hosted.weblate.org/widgets/librephotos/-/librephotos-frontend/svg-badge.svg" alt="Translation status" />
</a>

![LibrePhotos Screenshot](https://github.com/LibrePhotos/librephotos/blob/dev/screenshots/mockups_main_fhd.png?raw=true)
<sub>Mockup designed by rawpixel.com / Freepik</sub>

## Key Features of LibrePhotos

*   **Comprehensive Media Support:** Supports photos, including RAW files, and videos.
*   **Smart Organization:** Features a timeline view and automatically generates albums based on events.
*   **Advanced Search:** Includes face recognition, object/scene detection, semantic image search, and metadata search.
*   **Multi-User Support:** Allows multiple users to manage and share their media.
*   **File System Scanning:** Efficiently scans your file system to import photos and videos.

## Demos

*   **Stable Demo:** [https://demo1.librephotos.com/](https://demo1.librephotos.com/) (User: `demo`, Password: `demo1234`)
*   **Development Demo:** [https://demo2.librephotos.com/](https://demo2.librephotos.com/) (User: `demo`, Password: `demo1234`)

## Installation and Getting Started

Detailed installation instructions are available in our [documentation](https://docs.librephotos.com/docs/installation/standard-install).

## How to Contribute & Help

*   ⭐ **Star** the repository to show your support!
*   🚀 **Development:** Follow [this guide](https://docs.librephotos.com/docs/development/dev-install) to start developing in under 30 minutes.
*   🗒️ **Documentation:** Improve the documentation via a pull request [here](https://github.com/LibrePhotos/librephotos.docs).
*   🧪 **Testing:** Help find bugs by using the ```dev``` tag and reporting issues.
*   🧑‍🤝‍🧑 **Outreach:** Share LibrePhotos with others.
*   🌐 **Translations:** Make LibrePhotos accessible by translating it with [weblate](https://hosted.weblate.org/engage/librephotos/).
*   💸 **Donate:** Support the developers via [GitHub Sponsors](https://github.com/sponsors/derneuere).

## Technologies Used

LibrePhotos leverages several open-source technologies for its functionality:

*   **Image Conversion:** [ImageMagick](https://github.com/ImageMagick/ImageMagick)
*   **Video Conversion:** [FFmpeg](https://github.com/FFmpeg/FFmpeg)
*   **Exif Support:** [ExifTool](https://github.com/exiftool/exiftool)
*   **Face Detection:** [face\_recognition](https://github.com/ageitgey/face_recognition)
*   **Face Classification/Clusterization:** [scikit-learn](https://scikit-learn.org/) and [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
*   **Image Captioning:** [im2txt](https://github.com/HughKu/Im2txt),
*   **Scene Classification** [places365](http://places.csail.mit.edu/)
*   **Reverse Geocoding:** [Mapbox](https://www.mapbox.com/) (Requires an API key; first 50,000 geocode lookups are free per month)

[discord]: https://discord.gg/xwRvtSDGWb

For more information, visit the [LibrePhotos GitHub repository](https://github.com/LibrePhotos/librephotos).