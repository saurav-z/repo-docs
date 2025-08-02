[![Discord](https://img.shields.io/discord/784619049208250388?style=plastic)][discord] [![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&style=plastic&up_color=blue&up_message=online&url=https%3A%2F%2Flibrephotos.com)](https://librephotos.com/)
[![Read the docs](https://img.shields.io/static/v1?label=Read&message=the%20docs&color=blue&style=plastic)](https://docs.librephotos.com/) [![GitHub contributors](https://img.shields.io/github/contributors/librephotos/librephotos?style=plastic)](https://github.com/LibrePhotos/librephotos/graphs/contributors)
<a href="https://hosted.weblate.org/engage/librephotos/">
<img src="https://hosted.weblate.org/widgets/librephotos/-/librephotos-frontend/svg-badge.svg" alt="Translation status" />
</a>

# LibrePhotos: Your Self-Hosted, Open-Source Photo Management Solution

LibrePhotos is a powerful and privacy-focused open-source photo management system that gives you complete control over your photo library. [Visit the original repository](https://github.com/LibrePhotos/librephotos) to learn more.

[![](https://github.com/LibrePhotos/librephotos/blob/dev/screenshots/mockups_main_fhd.png?raw=true)](https://github.com/LibrePhotos/librephotos)
<sub>Mockup designed by rawpixel.com / Freepik</sub>

## Key Features of LibrePhotos

*   **Comprehensive Media Support:** Handles photos (including RAW formats) and videos.
*   **Organized Timeline View:** Easily browse your photos chronologically.
*   **Automatic Organization:** Scans your file system to organize your photos.
*   **Multi-User Support:** Share and manage your photo library with others.
*   **Intelligent Album Generation:** Creates albums based on events and locations (e.g., "Thursday in Berlin").
*   **Advanced Search Capabilities:**
    *   Face recognition and classification
    *   Reverse geocoding
    *   Object and scene detection
    *   Semantic image search (search by keywords)
    *   Search by metadata

## Demos

*   **Stable Demo:** [https://demo1.librephotos.com/](https://demo1.librephotos.com/) (User: `demo`, Password: `demo1234`)
*   **Development Demo:** [https://demo2.librephotos.com/](https://demo2.librephotos.com/) (User: `demo`, Password: `demo1234`)

## Installation

Detailed installation instructions are available in the official [documentation](https://docs.librephotos.com/docs/installation/standard-install).

## How to Contribute and Help Out

We welcome contributions from the community! Here's how you can help:

*   ⭐ **Star the Repository:** Show your support by starring the project on GitHub.
*   🚀 **Development:** Get started developing in less than 30 minutes using [this guide](https://docs.librephotos.com/docs/development/dev-install).
*   🗒️ **Documentation:** Improve the documentation by submitting pull requests [here](https://github.com/LibrePhotos/librephotos.docs).
*   🧪 **Testing:** Test the `dev` tag and report any bugs you find by opening an issue.
*   🧑‍🤝‍🧑 **Outreach:** Spread the word about LibrePhotos!
*   🌐 **Translations:** Help translate LibrePhotos into different languages via [weblate](https://hosted.weblate.org/engage/librephotos/).
*   💸 **Donate:** Support the developers through [GitHub Sponsors](https://github.com/sponsors/derneuere).

## Technologies Used

LibrePhotos leverages the following technologies:

*   **Image Conversion:** [ImageMagick](https://github.com/ImageMagick/ImageMagick)
*   **Video Conversion:** [FFmpeg](https://github.com/FFmpeg/FFmpeg)
*   **Exif Support:** [ExifTool](https://github.com/exiftool/exiftool)
*   **Face Detection:** [face_recognition](https://github.com/ageitgey/face_recognition)
*   **Face Classification/Clusterization:** [scikit-learn](https://scikit-learn.org/) and [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
*   **Image Captioning:** [im2txt](https://github.com/HughKu/Im2txt)
*   **Scene Classification:** [places365](http://places.csail.mit.edu/)
*   **Reverse Geocoding:** [Mapbox](https://www.mapbox.com/) (Requires an API key; first 50,000 geocode lookups are free per month)

[discord]: https://discord.gg/xwRvtSDGWb