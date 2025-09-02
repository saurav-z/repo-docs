[![Discord](https://img.shields.io/discord/784619049208250388?style=plastic)][discord]
[![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&style=plastic&up_color=blue&up_message=online&url=https%3A%2F%2Flibrephotos.com)](https://librephotos.com/)
[![Read the docs](https://img.shields.io/static/v1?label=Read&message=the%20docs&color=blue&style=plastic)](https://docs.librephotos.com/)
[![GitHub contributors](https://img.shields.io/github/contributors/librephotos/librephotos?style=plastic)](https://github.com/LibrePhotos/librephotos/graphs/contributors)
<a href="https://hosted.weblate.org/engage/librephotos/">
<img src="https://hosted.weblate.org/widgets/librephotos/-/librephotos-frontend/svg-badge.svg" alt="Translation status" />
</a>

# LibrePhotos: Your Self-Hosted, Open-Source Photo Management Solution

LibrePhotos is a powerful, open-source photo management solution designed for organizing, sharing, and backing up your photos and videos. For more information, visit the original repository: [LibrePhotos on GitHub](https://github.com/LibrePhotos/librephotos).

![LibrePhotos Screenshot](https://github.com/LibrePhotos/librephotos/blob/dev/screenshots/mockups_main_fhd.png?raw=true)
<sub>Mockup designed by rawpixel.com / Freepik</sub>

## Key Features

*   **Comprehensive Media Support:** Supports photos (including RAW files) and videos.
*   **Organized Timeline View:**  Easily browse your memories chronologically.
*   **Automated Organization:** Scans your file system and generates albums based on events (e.g., "Thursday in Berlin").
*   **Multi-User Support:** Share and manage your photos with others.
*   **Advanced Search:** Utilize face recognition, object/scene detection, and metadata-based search to find photos quickly.
*   **Face Recognition and Clustering:** Automatically identify and group faces for easy tagging.
*   **Location Awareness:** Reverse geocoding for location-based photo organization.
*   **Semantic Image Search:**  Find photos using natural language descriptions.

## Installation

Detailed installation instructions are available in our [documentation](https://docs.librephotos.com/docs/installation/standard-install).

*   **Demo Access:**
    *   **Stable Demo:** [https://demo1.librephotos.com/](https://demo1.librephotos.com/) - User: `demo`, Password: `demo1234` (with sample images).
    *   **Development Demo:** [https://demo2.librephotos.com/](https://demo2.librephotos.com/) - User: `demo`, Password: `demo1234`

## How to Contribute

We welcome contributions from the community!

*   ⭐ **Star** the repository to show your support!
*   🚀 **Development:**  Follow [this guide](https://docs.librephotos.com/docs/development/dev-install) to get started in under 30 minutes.
*   🗒️ **Documentation:**  Help improve our documentation by submitting pull requests [here](https://github.com/LibrePhotos/librephotos.docs).
*   🧪 **Testing:**  Test the `dev` tag and report any bugs by opening an issue.
*   🧑‍🤝‍🧑 **Outreach:**  Spread the word about LibrePhotos!
*   🌐 **Translations:**  Help make LibrePhotos accessible to a wider audience via [weblate](https://hosted.weblate.org/engage/librephotos/).
*   💸 **Donate:** Support the developers via [GitHub Sponsors](https://github.com/sponsors/derneuere).

## Technologies Used

LibrePhotos utilizes several technologies to provide its features:

*   **Image Conversion:** [ImageMagick](https://github.com/ImageMagick/ImageMagick)
*   **Video Conversion:** [FFmpeg](https://github.com/FFmpeg/FFmpeg)
*   **Exif Support:** [ExifTool](https://github.com/exiftool/exiftool)
*   **Face Detection:** [face_recognition](https://github.com/ageitgey/face_recognition)
*   **Face Classification/Clusterization:** [scikit-learn](https://scikit-learn.org/) and [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
*   **Image Captioning:** [im2txt](https://github.com/HughKu/Im2txt)
*   **Scene Classification:** [places365](http://places.csail.mit.edu/)
*   **Reverse Geocoding:** [Mapbox](https://www.mapbox.com/) (requires an API key; first 50,000 geocode lookups are free monthly)

[discord]: https://discord.gg/xwRvtSDGWb