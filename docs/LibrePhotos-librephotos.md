[![Discord](https://img.shields.io/discord/784619049208250388?style=plastic)][discord] [![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&style=plastic&up_color=blue&up_message=online&url=https%3A%2F%2Flibrephotos.com)](https://librephotos.com/)
[![Read the docs](https://img.shields.io/static/v1?label=Read&message=the%20docs&color=blue&style=plastic)](https://docs.librephotos.com/) [![GitHub contributors](https://img.shields.io/github/contributors/librephotos/librephotos?style=plastic)](https://github.com/LibrePhotos/librephotos/graphs/contributors)
<a href="https://hosted.weblate.org/engage/librephotos/">
<img src="https://hosted.weblate.org/widgets/librephotos/-/librephotos-frontend/svg-badge.svg" alt="Translation status" />
</a>

# LibrePhotos: Your Self-Hosted Photo Management Solution

**LibrePhotos** is a powerful, open-source photo management solution that lets you take control of your photos with a clean, modern interface. ([View the original repository](https://github.com/LibrePhotos/librephotos))

![LibrePhotos Screenshot](https://github.com/LibrePhotos/librephotos/blob/dev/screenshots/mockups_main_fhd.png?raw=true)
<sub>Mockup designed by rawpixel.com / Freepik</sub>

## Key Features of LibrePhotos

*   **Comprehensive Media Support:** Supports photos (including RAW) and videos.
*   **Organized Viewing:** Offers a timeline view for easy browsing.
*   **Automated Organization:** Scans your file system, generates albums based on events, and utilizes facial recognition.
*   **Intelligent Search:** Includes object/scene detection and semantic image search for easy photo finding.
*   **Multi-User Support:** Allows for multiple users to access and manage photos.
*   **Advanced Features:** Provides reverse geocoding for location-based organization.

## Demos

*   **Stable Demo:** [https://demo1.librephotos.com/](https://demo1.librephotos.com/). Username: `demo`, Password: `demo1234` (with sample images).
*   **Development Demo:** [https://demo2.librephotos.com/](https://demo2.librephotos.com/) (same user/password)

## Installation

Find detailed, step-by-step installation instructions in our [documentation](https://docs.librephotos.com/docs/installation/standard-install).

## How You Can Contribute

*   ⭐ **Star** the repository if you like the project!
*   🚀 **Develop:**  Get started developing in under 30 minutes following [this guide](https://docs.librephotos.com/docs/development/dev-install).
*   🗒️ **Document:** Improve the documentation by submitting a pull request [here](https://github.com/LibrePhotos/librephotos.docs).
*   🧪 **Test:** Help find bugs by using the ```dev``` tag and reporting any issues.
*   🧑‍🤝‍🧑 **Outreach:**  Share LibrePhotos with others and help them get started!
*   🌐 **Translate:** Make LibrePhotos accessible in more languages via [weblate](https://hosted.weblate.org/engage/librephotos/).
*   💸 **Donate:** Support the developers of LibrePhotos via [Github Sponsors](https://github.com/sponsors/derneuere).

## Technologies Used

*   **Image Conversion:** [ImageMagick](https://github.com/ImageMagick/ImageMagick)
*   **Video Conversion:** [FFmpeg](https://github.com/FFmpeg/FFmpeg)
*   **Exif Support:** [ExifTool](https://github.com/exiftool/exiftool)
*   **Face Detection:** [face_recognition](https://github.com/ageitgey/face_recognition)
*   **Face Classification/Clusterization:** [scikit-learn](https://scikit-learn.org/) and [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
*   **Image Captioning:** [im2txt](https://github.com/HughKu/Im2txt),
*   **Scene Classification:** [places365](http://places.csail.mit.edu/)
*   **Reverse Geocoding:** [Mapbox](https://www.mapbox.com/):  Requires an API key.  First 50,000 geocode lookups are free per month.

[discord]: https://discord.gg/xwRvtSDGWb