[![Discord](https://img.shields.io/discord/784619049208250388?style=plastic)][discord] [![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&style=plastic&up_color=blue&up_message=online&url=https%3A%2F%2Flibrephotos.com)](https://librephotos.com/)
[![Read the docs](https://img.shields.io/static/v1?label=Read&message=the%20docs&color=blue&style=plastic)](https://docs.librephotos.com/) [![GitHub contributors](https://img.shields.io/github/contributors/librephotos/librephotos?style=plastic)](https://github.com/LibrePhotos/librephotos/graphs/contributors)
<a href="https://hosted.weblate.org/engage/librephotos/">
<img src="https://hosted.weblate.org/widgets/librephotos/-/librephotos-frontend/svg-badge.svg" alt="Translation status" />
</a>

# LibrePhotos: Your Self-Hosted Photo Management Solution

**LibrePhotos** is a free and open-source photo management application that gives you complete control over your photo library. [Check out the original repository!](https://github.com/LibrePhotos/librephotos)

![LibrePhotos Screenshot](https://github.com/LibrePhotos/librephotos/blob/dev/screenshots/mockups_main_fhd.png?raw=true)
<sub>Mockup designed by rawpixel.com / Freepik</sub>

## Key Features

*   **Comprehensive Media Support:** Supports all photo types, including raw images, and videos.
*   **Intelligent Organization:**  Provides a timeline view, and automatically generates albums based on events (e.g., "Thursday in Berlin").
*   **Advanced Search Capabilities:**  Includes face recognition, object/scene detection, semantic image search, and metadata-based search for easy photo retrieval.
*   **Multi-User Support:** Manage your photo library collaboratively.
*   **Geotagging and Reverse Geocoding:** Maps your photos based on location data.
*   **User-Friendly Interface:** Offers a clean and intuitive experience for managing your photos.
*   **Open Source & Self-Hosted:** Complete control over your data.

## Get Started

*   **Demo:** Explore the stable demo: [https://demo1.librephotos.com/](https://demo1.librephotos.com/) (User: `demo`, Password: `demo1234`)
*   **Development Demo:** Try out the latest development version: [https://demo2.librephotos.com/](https://demo2.librephotos.com/) (User: `demo`, Password: `demo1234`)
*   **Installation:** Follow the step-by-step instructions in our [documentation](https://docs.librephotos.com/docs/installation/standard-install).

## How to Contribute

*   ⭐ **Star** the repository to show your support!
*   🚀 **Develop:** Get started quickly by following the [development guide](https://docs.librephotos.com/docs/development/dev-install).
*   🗒️ **Document:** Improve the documentation by submitting a pull request [here](https://github.com/LibrePhotos/librephotos.docs).
*   🧪 **Test:** Help find bugs by using the `dev` tag and reporting issues.
*   🧑‍🤝‍🧑 **Outreach:** Share LibrePhotos with others!
*   🌐 **Translate:** Make LibrePhotos accessible in more languages via [weblate](https://hosted.weblate.org/engage/librephotos/).
*   💸 **Donate:** Support the developers on [GitHub Sponsors](https://github.com/sponsors/derneuere).

## Technology Stack

LibrePhotos leverages several open-source libraries and tools:

*   **Image Conversion:** [ImageMagick](https://github.com/ImageMagick/ImageMagick)
*   **Video Conversion:** [FFmpeg](https://github.com/FFmpeg/FFmpeg)
*   **Exif Support:** [ExifTool](https://github.com/exiftool/exiftool)
*   **Face Detection:** [face\_recognition](https://github.com/ageitgey/face_recognition)
*   **Face Classification/Clustering:** [scikit-learn](https://scikit-learn.org/) and [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
*   **Image Captioning:** [im2txt](https://github.com/HughKu/Im2txt)
*   **Scene Classification:** [places365](http://places.csail.mit.edu/)
*   **Reverse Geocoding:** [Mapbox](https://www.mapbox.com/) (Requires an API key; first 50,000 lookups are free monthly)

[discord]: https://discord.gg/xwRvtSDGWb