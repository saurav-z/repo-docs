[![Discord](https://img.shields.io/discord/784619049208250388?style=plastic)][discord]
[![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&style=plastic&up_color=blue&up_message=online&url=https%3A%2F%2Flibrephotos.com)](https://librephotos.com/)
[![Read the docs](https://img.shields.io/static/v1?label=Read&message=the%20docs&color=blue&style=plastic)](https://docs.librephotos.com/)
[![GitHub contributors](https://img.shields.io/github/contributors/librephotos/librephotos?style=plastic)](https://github.com/LibrePhotos/librephotos/graphs/contributors)
<a href="https://hosted.weblate.org/engage/librephotos/">
<img src="https://hosted.weblate.org/widgets/librephotos/-/librephotos-frontend/svg-badge.svg" alt="Translation status" />
</a>

# LibrePhotos: Your Self-Hosted Photo Management Solution

Tired of vendor lock-in and privacy concerns with your photos? **LibrePhotos empowers you to take control of your memories with a powerful, open-source, self-hosted photo management solution.**

[View the source code on GitHub](https://github.com/LibrePhotos/librephotos)

![](https://github.com/LibrePhotos/librephotos/blob/dev/screenshots/mockups_main_fhd.png?raw=true)
<sub>Mockup designed by rawpixel.com / Freepik</sub>

## Key Features

*   ✅ **Supports all Photo Types:** Including RAW and video files.
*   🗓️ **Timeline View:** Organize your photos chronologically.
*   🔍 **Automated Organization:** Scans file systems, generates albums based on events.
*   👤 **Multi-User Support:** Allows for multiple users to manage their photos.
*   👤 **Face Recognition & Classification:** Automatically detects and organizes faces.
*   📍 **Reverse Geocoding:** Adds location data to your photos.
*   🖼️ **Object & Scene Detection:** Intelligent image analysis for better searchability.
*   🔎 **Semantic Image Search:** Find photos based on content descriptions.
*   🏷️ **Metadata Search:** Search by keywords and other metadata.

## Demo

*   **Stable Demo:** [https://demo1.librephotos.com/](https://demo1.librephotos.com/)
    *   User: `demo`, Password: `demo1234`
*   **Development Demo:** [https://demo2.librephotos.com/](https://demo2.librephotos.com/)
    *   User: `demo`, Password: `demo1234`

## Installation

Detailed installation instructions are available in the [LibrePhotos documentation](https://docs.librephotos.com/docs/installation/standard-install).

## How to Contribute

*   ⭐ **Star** the repository to show your support!
*   🚀 **Develop:** Follow the [development guide](https://docs.librephotos.com/docs/development/dev-install) to get started in under 30 minutes.
*   ✍️ **Document:** Improve the documentation by submitting pull requests [here](https://github.com/LibrePhotos/librephotos.docs).
*   🧪 **Test:** Help find bugs by using the `dev` tag and reporting issues.
*   📣 **Outreach:** Share LibrePhotos with others!
*   🌐 **Translate:** Make LibrePhotos accessible to more people with [weblate](https://hosted.weblate.org/engage/librephotos/).
*   💸 **Donate:** Support the project developers with a donation on [GitHub Sponsors](https://github.com/sponsors/derneuere).

## Technologies Used

*   **Image Conversion:** [ImageMagick](https://github.com/ImageMagick/ImageMagick)
*   **Video Conversion:** [FFmpeg](https://github.com/FFmpeg/FFmpeg)
*   **Exif Support:** [ExifTool](https://github.com/exiftool/exiftool)
*   **Face Detection:** [face_recognition](https://github.com/ageitgey/face_recognition)
*   **Face Classification/Clustering:** [scikit-learn](https://scikit-learn.org/) and [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
*   **Image Captioning:** [im2txt](https://github.com/HughKu/Im2txt)
*   **Scene Classification:** [places365](http://places.csail.mit.edu/)
*   **Reverse Geocoding:** [Mapbox](https://www.mapbox.com/) (API key required; free tier available)

[discord]: https://discord.gg/xwRvtSDGWb