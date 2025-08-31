[![Discord](https://img.shields.io/discord/784619049208250388?style=plastic)][discord] [![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&style=plastic&up_color=blue&up_message=online&url=https%3A%2F%2Flibrephotos.com)](https://librephotos.com/)
[![Read the docs](https://img.shields.io/static/v1?label=Read&message=the%20docs&color=blue&style=plastic)](https://docs.librephotos.com/) [![GitHub contributors](https://img.shields.io/github/contributors/librephotos/librephotos?style=plastic)](https://github.com/LibrePhotos/librephotos/graphs/contributors)
<a href="https://hosted.weblate.org/engage/librephotos/">
<img src="https://hosted.weblate.org/widgets/librephotos/-/librephotos-frontend/svg-badge.svg" alt="Translation status" />
</a>

# LibrePhotos: Your Self-Hosted, Open-Source Photo Management Solution

**LibrePhotos** offers a powerful and private way to manage and enjoy your photo and video collection. ([View the original repository](https://github.com/LibrePhotos/librephotos))

## Key Features

*   **Comprehensive Media Support:** Handles photos (including RAW files) and videos.
*   **Intelligent Organization:** Timeline view, album generation based on events, and metadata-based search.
*   **Advanced AI Capabilities:** Face recognition and classification, object/scene detection, semantic image search.
*   **Multi-User Support:** Share and manage your photos with others.
*   **Geotagging:** Reverse geocoding to automatically add location data.

## Demo

*   **Stable Demo:** [https://demo1.librephotos.com/](https://demo1.librephotos.com/) (User: `demo`, Password: `demo1234`)
*   **Development Demo:** [https://demo2.librephotos.com/](https://demo2.librephotos.com/) (User: `demo`, Password: `demo1234`)

## Installation

Detailed installation instructions can be found in our comprehensive [documentation](https://docs.librephotos.com/docs/installation/standard-install).

## How to Contribute and Support

*   ⭐ **Star** the repository to show your support!
*   🚀 **Development:** Get started by following [this guide](https://docs.librephotos.com/docs/development/dev-install).
*   🗒️ **Documentation:** Improve the documentation by submitting a pull request [here](https://github.com/LibrePhotos/librephotos.docs).
*   🧪 **Testing:** Help us find bugs by using the `dev` tag and reporting issues.
*   🧑‍🤝‍🧑 **Outreach:** Share LibrePhotos with others!
*   🌐 **Translations:** Contribute to making LibrePhotos accessible in more languages through [weblate](https://hosted.weblate.org/engage/librephotos/).
*   💸 **Donate:** Support the developers of LibrePhotos [here](https://github.com/sponsors/derneuere).

## Technologies Used

LibrePhotos leverages a robust stack of open-source technologies:

*   **Image Conversion:** [ImageMagick](https://github.com/ImageMagick/ImageMagick)
*   **Video Conversion:** [FFmpeg](https://github.com/FFmpeg/FFmpeg)
*   **Exif Support:** [ExifTool](https://github.com/exiftool/exiftool)
*   **Face Detection:** [face\_recognition](https://github.com/ageitgey/face_recognition)
*   **Face Classification/Clusterization:** [scikit-learn](https://scikit-learn.org/) and [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
*   **Image Captioning:** [im2txt](https://github.com/HughKu/Im2txt)
*   **Scene Classification:** [places365](http://places.csail.mit.edu/)
*   **Reverse Geocoding:** [Mapbox](https://www.mapbox.com/) (Requires an API key. First 50,000 geocode lookups are free per month.)

[discord]: https://discord.gg/xwRvtSDGWb