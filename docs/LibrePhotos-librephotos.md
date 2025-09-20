[![Discord](https://img.shields.io/discord/784619049208250388?style=plastic)][discord] [![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&style=plastic&up_color=blue&up_message=online&url=https%3A%2F%2Flibrephotos.com)](https://librephotos.com/)
[![Read the docs](https://img.shields.io/static/v1?label=Read&message=the%20docs&color=blue&style=plastic)](https://docs.librephotos.com/) [![GitHub contributors](https://img.shields.io/github/contributors/librephotos/librephotos?style=plastic)](https://github.com/LibrePhotos/librephotos/graphs/contributors)
<a href="https://hosted.weblate.org/engage/librephotos/">
<img src="https://hosted.weblate.org/widgets/librephotos/-/librephotos-frontend/svg-badge.svg" alt="Translation status" />
</a>

# LibrePhotos: Self-Hosted Photo Management and Face Recognition 

**Tired of relying on cloud storage for your photos?** LibrePhotos offers a powerful and open-source solution to manage, organize, and search your photo and video library with advanced features.

[Visit the original repository on GitHub](https://github.com/LibrePhotos/librephotos)

![LibrePhotos Interface](https://github.com/LibrePhotos/librephotos/blob/dev/screenshots/mockups_main_fhd.png?raw=true)
<sub>Mockup designed by rawpixel.com / Freepik</sub>

## Key Features of LibrePhotos

*   **Comprehensive Media Support:** Supports photos (including RAW formats) and videos.
*   **Intelligent Organization:**
    *   Timeline view for chronological browsing.
    *   Automatic album generation based on events (e.g., "Thursday in Berlin").
    *   Scans photos directly from your file system.
*   **Advanced Search & Discovery:**
    *   Face recognition and classification.
    *   Object and scene detection.
    *   Semantic image search.
    *   Search by metadata.
*   **Multi-User Collaboration:** Supports multiple users for shared photo libraries.
*   **Reverse Geocoding:** Maps photos based on their location data.

## Demo Access

*   **Stable Demo:** [https://demo1.librephotos.com/](https://demo1.librephotos.com/) - User: `demo`, Password: `demo1234` (with sample images).
*   **Development Demo:** [https://demo2.librephotos.com/](https://demo2.librephotos.com/) - User: `demo`, Password: `demo1234`

## How to Get Started

Detailed installation instructions are available in the [LibrePhotos Documentation](https://docs.librephotos.com/docs/installation/standard-install).

## How to Contribute

Help improve LibrePhotos and make it accessible to everyone!

*   ⭐ **Star** the repository on GitHub.
*   🚀 **Development:** Follow the [development guide](https://docs.librephotos.com/docs/development/dev-install) to get started.
*   🗒️ **Documentation:** Contribute to the documentation by submitting pull requests [here](https://github.com/LibrePhotos/librephotos.docs).
*   🧪 **Testing:** Test the `dev` tag and report any bugs by opening an issue.
*   🧑‍🤝‍🧑 **Outreach:** Share LibrePhotos with others and encourage them to join the community.
*   🌐 **Translations:** Translate LibrePhotos into more languages using [weblate](https://hosted.weblate.org/engage/librephotos/).
*   💸 [**Donate**](https://github.com/sponsors/derneuere) to support the developers of LibrePhotos.

## Technologies Used

LibrePhotos leverages several powerful libraries:

*   **Image Conversion:** [ImageMagick](https://github.com/ImageMagick/ImageMagick)
*   **Video Conversion:** [FFmpeg](https://github.com/FFmpeg/FFmpeg)
*   **Exif Support:** [ExifTool](https://github.com/exiftool/exiftool)
*   **Face Detection:** [face\_recognition](https://github.com/ageitgey/face_recognition)
*   **Face Classification/Clusterization:** [scikit-learn](https://scikit-learn.org/) and [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
*   **Image Captioning:** [im2txt](https://github.com/HughKu/Im2txt),
*   **Scene Classification:** [places365](http://places.csail.mit.edu/)
*   **Reverse Geocoding:** [geopy](https://github.com/geopy/geopy)

[discord]: https://discord.gg/xwRvtSDGWb