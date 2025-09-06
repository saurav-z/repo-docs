[![Discord](https://img.shields.io/discord/784619049208250388?style=plastic)][discord] [![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&style=plastic&up_color=blue&up_message=online&url=https%3A%2F%2Flibrephotos.com)](https://librephotos.com/)
[![Read the docs](https://img.shields.io/static/v1?label=Read&message=the%20docs&color=blue&style=plastic)](https://docs.librephotos.com/) [![GitHub contributors](https://img.shields.io/github/contributors/librephotos/librephotos?style=plastic)](https://github.com/LibrePhotos/librephotos/graphs/contributors)
<a href="https://hosted.weblate.org/engage/librephotos/">
<img src="https://hosted.weblate.org/widgets/librephotos/-/librephotos-frontend/svg-badge.svg" alt="Translation status" />
</a>

# LibrePhotos: Your Self-Hosted, Open-Source Photo Management Solution

**LibrePhotos is a powerful and privacy-focused, open-source alternative to Google Photos, giving you complete control over your photos and videos.** ([View on GitHub](https://github.com/LibrePhotos/librephotos))

![LibrePhotos Screenshot](https://github.com/LibrePhotos/librephotos/blob/dev/screenshots/mockups_main_fhd.png?raw=true)
<sub>Mockup designed by rawpixel.com / Freepik</sub>

## Key Features

*   **Comprehensive Media Support:**  Supports all photo types, including RAW files, and videos.
*   **Intelligent Organization:**
    *   Timeline View for easy browsing.
    *   Automatic album generation based on events and locations.
    *   Scans pictures on the file system.
*   **Advanced Search & Discovery:**
    *   Face recognition and classification.
    *   Object and scene detection.
    *   Semantic image search.
    *   Search by metadata.
*   **Multi-User Support:**  Share and manage your photos with others.
*   **Reverse Geocoding:** Automatically tags photos with location data.

## Installation

Detailed installation instructions are available in the [LibrePhotos documentation](https://docs.librephotos.com/docs/installation/standard-install).

## How to Contribute & Get Involved

*   ⭐ **Star** the repository on GitHub!
*   🚀 **Development:**  Get started with development in under 30 minutes using [this guide](https://docs.librephotos.com/docs/development/dev-install).
*   🗒️ **Documentation:** Improve the documentation by submitting pull requests to [this repository](https://github.com/LibrePhotos/librephotos.docs).
*   🧪 **Testing:**  Help find bugs by using the `dev` tag and reporting issues.
*   🧑‍🤝‍🧑 **Outreach:**  Spread the word about LibrePhotos!
*   🌐 **Translations:**  Make LibrePhotos accessible to more people through [Weblate](https://hosted.weblate.org/engage/librephotos/).
*   💸 **Donate:**  Support the developers via [GitHub Sponsors](https://github.com/sponsors/derneuere).

## Technologies Used

LibrePhotos leverages several key technologies:

*   **Image Conversion:** [ImageMagick](https://github.com/ImageMagick/ImageMagick)
*   **Video Conversion:** [FFmpeg](https://github.com/FFmpeg/FFmpeg)
*   **EXIF Support:** [ExifTool](https://github.com/exiftool/exiftool)
*   **Face Detection:** [face_recognition](https://github.com/ageitgey/face_recognition)
*   **Face Classification/Clustering:** [scikit-learn](https://scikit-learn.org/) and [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
*   **Image Captioning:** [im2txt](https://github.com/HughKu/Im2txt)
*   **Scene Classification:** [places365](http://places.csail.mit.edu/)
*   **Reverse Geocoding:** [Mapbox](https://www.mapbox.com/) (requires an API key - first 50,000 geocode lookups are free per month)

[discord]: https://discord.gg/xwRvtSDGWb