[![Discord](https://img.shields.io/discord/784619049208250388?style=plastic)][discord]
[![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&style=plastic&up_color=blue&up_message=online&url=https%3A%2F%2Flibrephotos.com)](https://librephotos.com/)
[![Read the docs](https://img.shields.io/static/v1?label=Read&message=the%20docs&color=blue&style=plastic)](https://docs.librephotos.com/)
[![GitHub contributors](https://img.shields.io/github/contributors/librephotos/librephotos?style=plastic)](https://github.com/LibrePhotos/librephotos/graphs/contributors)
<a href="https://hosted.weblate.org/engage/librephotos/">
<img src="https://hosted.weblate.org/widgets/librephotos/-/librephotos-frontend/svg-badge.svg" alt="Translation status" />
</a>

# LibrePhotos: Your Self-Hosted, Open-Source Photo Management Solution

LibrePhotos is a powerful, open-source photo management application that lets you take control of your photo library.  ([View the original repository](https://github.com/LibrePhotos/librephotos))

## Key Features

*   **Comprehensive Media Support:** Supports photos (including RAW files) and videos.
*   **Intelligent Organization:**
    *   Timeline view for easy browsing.
    *   Automatic album generation based on events.
    *   Scans your file system for quick import.
*   **Advanced Search & Discovery:**
    *   Face recognition and classification for easy people tagging.
    *   Semantic image search to find photos based on content.
    *   Object and scene detection.
    *   Search by metadata.
*   **Multi-User Support:** Share your photos with others.
*   **Location-Aware:** Reverse geocoding to map your photos.

## Getting Started

*   **Live Demos:**
    *   **Stable Demo:** [https://demo1.librephotos.com/](https://demo1.librephotos.com/) (User: `demo`, Password: `demo1234`)
    *   **Development Demo:** [https://demo2.librephotos.com/](https://demo2.librephotos.com/) (User: `demo`, Password: `demo1234`)

*   **Installation:** Detailed instructions are available in our [documentation](https://docs.librephotos.com/docs/installation/standard-install).

## How to Contribute

We welcome contributions from the community!

*   ⭐ **Star** the repository if you like this project!
*   🚀 **Developing:** Follow the [development guide](https://docs.librephotos.com/docs/development/dev-install) to get started in minutes.
*   🗒️ **Documentation:** Help improve our documentation by submitting pull requests [here](https://github.com/LibrePhotos/librephotos.docs).
*   🧪 **Testing:** Use the `dev` tag and report any bugs by opening an issue.
*   🧑‍🤝‍🧑 **Outreach:** Spread the word about LibrePhotos!
*   🌐 **Translations:** Make LibrePhotos accessible to more people with [weblate](https://hosted.weblate.org/engage/librephotos/).
*   💸 [**Donate**](https://github.com/sponsors/derneuere) to the developers of LibrePhotos

## Technologies Used

LibrePhotos leverages the following technologies:

*   **Image Conversion:** [ImageMagick](https://github.com/ImageMagick/ImageMagick)
*   **Video Conversion:** [FFmpeg](https://github.com/FFmpeg/FFmpeg)
*   **Exif Support:** [ExifTool](https://github.com/exiftool/exiftool)
*   **Face Detection:** [face\_recognition](https://github.com/ageitgey/face_recognition)
*   **Face Classification/Clusterization:** [scikit-learn](https://scikit-learn.org/) and [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
*   **Image Captioning:** [im2txt](https://github.com/HughKu/Im2txt)
*   **Scene Classification:** [places365](http://places.csail.mit.edu/)
*   **Reverse Geocoding:** [Mapbox](https://www.mapbox.com/) (Requires an API key; the first 50,000 geocode lookups are free monthly)

[discord]: https://discord.gg/xwRvtSDGWb