[![Discord](https://img.shields.io/discord/784619049208250388?style=plastic)][discord] [![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&style=plastic&up_color=blue&up_message=online&url=https%3A%2F%2Flibrephotos.com)](https://librephotos.com/)
[![Read the docs](https://img.shields.io/static/v1?label=Read&message=the%20docs&color=blue&style=plastic)](https://docs.librephotos.com/) [![GitHub contributors](https://img.shields.io/github/contributors/librephotos/librephotos?style=plastic)](https://github.com/LibrePhotos/librephotos/graphs/contributors)
<a href="https://hosted.weblate.org/engage/librephotos/">
<img src="https://hosted.weblate.org/widgets/librephotos/-/librephotos-frontend/svg-badge.svg" alt="Translation status" />
</a>

# LibrePhotos: Your Self-Hosted Photo Management Solution

**Tired of relying on cloud services? LibrePhotos offers a powerful, open-source alternative for managing and organizing your photos and videos.**  [Check out the original repository](https://github.com/LibrePhotos/librephotos)

![LibrePhotos Screenshot](https://github.com/LibrePhotos/librephotos/blob/dev/screenshots/mockups_main_fhd.png?raw=true)
<sub>Mockup designed by rawpixel.com / Freepik</sub>

## Key Features

*   **Comprehensive Media Support:** Supports all photo types, including RAW files, and videos.
*   **Intelligent Organization:** Automatically scans your file system and generates albums based on events (e.g., "Thursday in Berlin").
*   **Advanced Search:** Utilize face recognition, object/scene detection, and semantic image search for effortless photo retrieval.
*   **Multiuser Support:**  Share your photos with friends and family.
*   **Interactive Timeline View:** Browse your photos and videos chronologically.
*   **Face Recognition & Clustering:** Automatically identify and group faces for easy organization.
*   **Geotagging & Reverse Geocoding:** Automatically locate your photos on a map.

## Installation

Detailed installation instructions are available in our [documentation](https://docs.librephotos.com/docs/installation/standard-install).

## Demos
*   **Stable Demo:** [https://demo1.librephotos.com/](https://demo1.librephotos.com/). User: `demo`, Password: `demo1234`
*   **Development Demo:** [https://demo2.librephotos.com/](https://demo2.librephotos.com/). User: `demo`, Password: `demo1234`

## How to Contribute

We welcome contributions from the community!

*   ⭐ **Star** the repository to show your support!
*   🚀 **Develop:** Get started quickly with our [development guide](https://docs.librephotos.com/docs/development/dev-install).
*   🗒️ **Documentation:**  Improve the documentation by submitting a pull request [here](https://github.com/LibrePhotos/librephotos.docs).
*   🧪 **Testing:** Help us find bugs by using the `dev` tag and reporting issues.
*   🧑‍🤝‍🧑 **Outreach:** Spread the word about LibrePhotos!
*   🌐 **Translations:** Make LibrePhotos accessible to more users through [Weblate](https://hosted.weblate.org/engage/librephotos/).
*   💸 [**Donate**](https://github.com/sponsors/derneuere) to support the developers.

## Technologies Used

LibrePhotos leverages several open-source technologies:

*   **Image Conversion:** [ImageMagick](https://github.com/ImageMagick/ImageMagick)
*   **Video Conversion:** [FFmpeg](https://github.com/FFmpeg/FFmpeg)
*   **Exif Support:** [ExifTool](https://github.com/exiftool/exiftool)
*   **Face Detection:** [face\_recognition](https://github.com/ageitgey/face_recognition)
*   **Face Classification/Clustering:** [scikit-learn](https://scikit-learn.org/) and [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
*   **Image Captioning:** [im2txt](https://github.com/HughKu/Im2txt)
*   **Scene Classification:** [places365](http://places.csail.mit.edu/)
*   **Reverse Geocoding:** [geopy](https://github.com/geopy/geopy)

[discord]: https://discord.gg/xwRvtSDGWb