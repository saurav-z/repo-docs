# LibrePhotos: Your Self-Hosted Photo Management Solution

**Tired of relying on cloud services for your photos?** LibrePhotos offers a powerful, open-source solution for self-hosting and managing your entire photo and video library, putting you in complete control.

[![Discord](https://img.shields.io/discord/784619049208250388?style=plastic)][discord] [![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&style=plastic&up_color=blue&up_message=online&url=https%3A%2F%2Flibrephotos.com)](https://librephotos.com/)
[![Read the docs](https://img.shields.io/static/v1?label=Read&message=the%20docs&color=blue&style=plastic)](https://docs.librephotos.com/) [![GitHub contributors](https://img.shields.io/github/contributors/librephotos/librephotos?style=plastic)](https://github.com/LibrePhotos/librephotos/graphs/contributors)
<a href="https://hosted.weblate.org/engage/librephotos/">
<img src="https://hosted.weblate.org/widgets/librephotos/-/librephotos-frontend/svg-badge.svg" alt="Translation status" />
</a>

## Key Features

*   **Comprehensive Media Support:** Supports all major photo formats, including RAW, and video files.
*   **Organized Timeline View:** Browse your photos chronologically for easy access.
*   **Automated Organization:** Scans your file system to find and import photos.
*   **Multi-User Support:** Allows multiple users with their own accounts.
*   **Intelligent Album Creation:** Automatically generates albums based on events and locations.
*   **Advanced AI Features:** Includes face recognition, scene detection, and semantic image search for powerful organization and discovery.
*   **Metadata-Based Search:** Search your photos using a variety of metadata tags.
*   **Reverse Geocoding:** Automatically identifies locations for your photos using geolocation data.

## Demo

*   **Stable Demo:** [https://demo1.librephotos.com/](https://demo1.librephotos.com/) - User: `demo`, Password: `demo1234` (sample images).
*   **Development Demo:** [https://demo2.librephotos.com/](https://demo2.librephotos.com/) - User: `demo`, Password: `demo1234` (same user/password).

## Installation

Detailed installation instructions are available in our [documentation](https://docs.librephotos.com/docs/installation/standard-install).

## How to Contribute & Help Out

*   ⭐ **Star** the repository on [GitHub](https://github.com/LibrePhotos/librephotos) if you like the project!
*   🚀 **Development:** Get started by following [this guide](https://docs.librephotos.com/docs/development/dev-install).
*   🗒️ **Documentation:** Improve the documentation through pull requests [here](https://github.com/LibrePhotos/librephotos.docs).
*   🧪 **Testing:** Help identify bugs by using the `dev` tag and reporting issues.
*   🧑‍🤝‍🧑 **Outreach:** Share LibrePhotos with others!
*   🌐 **Translations:** Help make LibrePhotos accessible to more people on [weblate](https://hosted.weblate.org/engage/librephotos/).
*   💸 [**Donate**](https://github.com/sponsors/derneuere) to support the developers.

## Technologies Used

*   **Image Conversion:** [ImageMagick](https://github.com/ImageMagick/ImageMagick)
*   **Video Conversion:** [FFmpeg](https://github.com/FFmpeg/FFmpeg)
*   **Exif Support:** [ExifTool](https://github.com/exiftool/exiftool)
*   **Face Detection:** [face_recognition](https://github.com/ageitgey/face_recognition)
*   **Face Classification/Clusterization:** [scikit-learn](https://scikit-learn.org/) and [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
*   **Image Captioning:** [im2txt](https://github.com/HughKu/Im2txt)
*   **Scene Classification:** [places365](http://places.csail.mit.edu/)
*   **Reverse Geocoding:** [Mapbox](https://www.mapbox.com/): You need to have an API key. First 50,000 geocode lookups are free every month.

[discord]: https://discord.gg/xwRvtSDGWb