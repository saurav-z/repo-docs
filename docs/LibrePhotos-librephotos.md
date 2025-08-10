# LibrePhotos: Your Self-Hosted Photo Management Solution

LibrePhotos is a powerful, open-source photo management platform designed to help you organize, store, and share your photos and videos easily and securely. ([Original Repository](https://github.com/LibrePhotos/librephotos))

[![Discord](https://img.shields.io/discord/784619049208250388?style=plastic)][discord] [![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&style=plastic&up_color=blue&up_message=online&url=https%3A%2F%2Flibrephotos.com)](https://librephotos.com/)
[![Read the docs](https://img.shields.io/static/v1?label=Read&message=the%20docs&color=blue&style=plastic)](https://docs.librephotos.com/) [![GitHub contributors](https://img.shields.io/github/contributors/librephotos/librephotos?style=plastic)](https://github.com/LibrePhotos/librephotos/graphs/contributors)
<a href="https://hosted.weblate.org/engage/librephotos/">
<img src="https://hosted.weblate.org/widgets/librephotos/-/librephotos-frontend/svg-badge.svg" alt="Translation status" />
</a>

![LibrePhotos Screenshot](https://github.com/LibrePhotos/librephotos/blob/dev/screenshots/mockups_main_fhd.png?raw=true)
<sub>Mockup designed by rawpixel.com / Freepik</sub>

## Key Features

*   **Comprehensive Media Support:** Supports photos (including RAW files) and videos.
*   **Organized Viewing:** Timeline view for easy browsing.
*   **Automated Organization:** Scans and organizes pictures from your file system.
*   **Multi-User Support:**  Allows multiple users to access and manage photos.
*   **Intelligent Album Creation:** Generates albums based on events like "Thursday in Berlin."
*   **Advanced Search:** Semantic image search and metadata search.
*   **AI-Powered Features:**
    *   Face recognition and classification.
    *   Object and scene detection.
    *   Reverse geocoding.

## Demos

*   **Stable Demo:** [https://demo1.librephotos.com/](https://demo1.librephotos.com/). User: `demo`, Password: `demo1234`
*   **Development Demo:** [https://demo2.librephotos.com/](https://demo2.librephotos.com/) (same user/password)

## Getting Started

Complete installation instructions are available in the [documentation](https://docs.librephotos.com/docs/installation/standard-install).

## How to Contribute

We welcome contributions of all kinds!

*   ⭐ **Star** the repository.
*   🚀 **Development:** Follow the [development guide](https://docs.librephotos.com/docs/development/dev-install) to get started.
*   🗒️ **Documentation:** Help improve the documentation by submitting pull requests [here](https://github.com/LibrePhotos/librephotos.docs).
*   🧪 **Testing:** Help us find bugs by using the `dev` tag and reporting issues.
*   🧑‍🤝‍🧑 **Outreach:** Spread the word about LibrePhotos!
*   🌐 **Translations:** Make LibrePhotos accessible in more languages with [weblate](https://hosted.weblate.org/engage/librephotos/).
*   💸 **Donate:** Support the project's developers [here](https://github.com/sponsors/derneuere).

## Technologies Used

*   **Image Conversion:** [ImageMagick](https://github.com/ImageMagick/ImageMagick)
*   **Video Conversion:** [FFmpeg](https://github.com/FFmpeg/FFmpeg)
*   **Exif Support:** [ExifTool](https://github.com/exiftool/exiftool)
*   **Face Detection:** [face_recognition](https://github.com/ageitgey/face_recognition)
*   **Face Classification/Clusterization:** [scikit-learn](https://scikit-learn.org/) and [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
*   **Image Captioning:** [im2txt](https://github.com/HughKu/Im2txt),
*   **Scene Classification:** [places365](http://places.csail.mit.edu/)
*   **Reverse Geocoding:** [Mapbox](https://www.mapbox.com/) (API key required; first 50,000 lookups are free per month)

[discord]: https://discord.gg/xwRvtSDGWb