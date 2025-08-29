[![Discord](https://img.shields.io/discord/784619049208250388?style=plastic)][discord] [![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&style=plastic&up_color=blue&up_message=online&url=https%3A%2F%2Flibrephotos.com)](https://librephotos.com/)
[![Read the docs](https://img.shields.io/static/v1?label=Read&message=the%20docs&color=blue&style=plastic)](https://docs.librephotos.com/) [![GitHub contributors](https://img.shields.io/github/contributors/librephotos/librephotos?style=plastic)](https://github.com/LibrePhotos/librephotos/graphs/contributors)
<a href="https://hosted.weblate.org/engage/librephotos/">
<img src="https://hosted.weblate.org/widgets/librephotos/-/librephotos-frontend/svg-badge.svg" alt="Translation status" />
</a>

# LibrePhotos: Your Self-Hosted, Open-Source Photo Management Solution

Tired of relying on proprietary photo services? **LibrePhotos** offers a powerful, open-source alternative for organizing, managing, and enjoying your entire photo and video library, putting you in complete control.  [View the source on GitHub](https://github.com/LibrePhotos/librephotos).

![LibrePhotos Screenshot](https://github.com/LibrePhotos/librephotos/blob/dev/screenshots/mockups_main_fhd.png?raw=true)
<sub>Mockup designed by rawpixel.com / Freepik</sub>

## Key Features

*   **Comprehensive Media Support:**  Handles photos of all types, including RAW files, and videos.
*   **Intelligent Organization:**  Timeline view, automatic album generation based on events and location, and smart search capabilities.
*   **Advanced AI Capabilities:** Includes face recognition and classification, object/scene detection, and semantic image search.
*   **Multi-User Support:**  Share and manage your photos with others.
*   **Robust Metadata Handling:**  Search and organize based on metadata.
*   **Reverse Geocoding:**  Automatically adds location data to your photos.

## Demos

*   **Stable Demo:** [https://demo1.librephotos.com/](https://demo1.librephotos.com/) (User: `demo`, Password: `demo1234`)
*   **Development Demo:** [https://demo2.librephotos.com/](https://demo2.librephotos.com/) (User: `demo`, Password: `demo1234`)

## Installation

Detailed installation instructions are available in our [documentation](https://docs.librephotos.com/docs/installation/standard-install).

## How to Contribute & Get Involved

We welcome contributions from everyone! Here's how you can help:

*   ⭐ **Star** the repository to show your support!
*   🚀 **Developing**:  Follow [this guide](https://docs.librephotos.com/docs/development/dev-install) to get started quickly.
*   🗒️ **Documentation**:  Improve the documentation by submitting a pull request [here](https://github.com/LibrePhotos/librephotos.docs).
*   🧪 **Testing**:  Help us find bugs using the `dev` tag and reporting any issues.
*   🧑‍🤝‍🧑 **Outreach**:  Spread the word about LibrePhotos!
*   🌐 **Translations**:  Make LibrePhotos accessible in more languages via [weblate](https://hosted.weblate.org/engage/librephotos/).
*   💸 [**Donate**](https://github.com/sponsors/derneuere) to support the developers.

## Technologies Used

LibrePhotos leverages the following open-source technologies:

*   **Image Conversion:** [ImageMagick](https://github.com/ImageMagick/ImageMagick)
*   **Video Conversion:** [FFmpeg](https://github.com/FFmpeg/FFmpeg)
*   **Exif Support:** [ExifTool](https://github.com/exiftool/exiftool)
*   **Face Detection:** [face\_recognition](https://github.com/ageitgey/face_recognition)
*   **Face Classification/Clusterization:** [scikit-learn](https://scikit-learn.org/) and [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
*   **Image Captioning:** [im2txt](https://github.com/HughKu/Im2txt)
*   **Scene Classification**: [places365](http://places.csail.mit.edu/)
*   **Reverse Geocoding:** [Mapbox](https://www.mapbox.com/) (API key required, free for the first 50,000 geocode lookups per month)

[discord]: https://discord.gg/xwRvtSDGWb