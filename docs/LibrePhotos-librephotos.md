[![Discord](https://img.shields.io/discord/784619049208250388?style=plastic)][discord] [![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&style=plastic&up_color=blue&up_message=online&url=https%3A%2F%2Flibrephotos.com)](https://librephotos.com/)
[![Read the docs](https://img.shields.io/static/v1?label=Read&message=the%20docs&color=blue&style=plastic)](https://docs.librephotos.com/) [![GitHub contributors](https://img.shields.io/github/contributors/librephotos/librephotos?style=plastic)](https://github.com/LibrePhotos/librephotos/graphs/contributors)
<a href="https://hosted.weblate.org/engage/librephotos/">
<img src="https://hosted.weblate.org/widgets/librephotos/-/librephotos-frontend/svg-badge.svg" alt="Translation status" />
</a>

# LibrePhotos: Your Self-Hosted Photo Management Solution

**LibrePhotos** is a powerful, open-source photo management solution that empowers you to organize, share, and enjoy your memories with complete control.  ([See the original repository](https://github.com/LibrePhotos/librephotos))

## Key Features

*   🖼️ **Comprehensive Media Support:** Supports a wide range of photo and video formats, including RAW images.
*   📅 **Intelligent Organization:** Organize your photos with a timeline view, event-based album generation, and metadata search.
*   👤 **Advanced Recognition:** Utilize face recognition, object detection, and scene classification for smart tagging and search.
*   🌍 **Location-Aware:** Leverage reverse geocoding to automatically tag photos with their location.
*   👨‍👩‍👧‍👦 **Multi-User Support:** Easily share your photo library with family and friends.
*   🔍 **Semantic Search:** Find photos effortlessly using natural language descriptions.

## Demo

*   **Stable Demo:** [https://demo1.librephotos.com/](https://demo1.librephotos.com/) (User: `demo`, Password: `demo1234`)
*   **Development Demo:** [https://demo2.librephotos.com/](https://demo2.librephotos.com/) (User: `demo`, Password: `demo1234`)

## Getting Started

Detailed installation instructions are available in our comprehensive [documentation](https://docs.librephotos.com/docs/installation/standard-install).

## How You Can Contribute

We welcome contributions from the community!  Here's how you can help:

*   ⭐ **Star** the repository on GitHub!
*   🚀 **Development:** Follow the [development guide](https://docs.librephotos.com/docs/development/dev-install) to get started in under 30 minutes.
*   🗒️ **Documentation:** Improve the documentation by submitting pull requests [here](https://github.com/LibrePhotos/librephotos.docs).
*   🧪 **Testing:** Help find bugs by using the `dev` tag and reporting issues.
*   🧑‍🤝‍🧑 **Outreach:** Spread the word about LibrePhotos!
*   🌐 **Translations:** Contribute to translations via [Weblate](https://hosted.weblate.org/engage/librephotos/).
*   💸 [**Donate**](https://github.com/sponsors/derneuere) to support the developers.

## Technologies Used

LibrePhotos leverages several powerful open-source tools:

*   **Image Conversion:** [ImageMagick](https://github.com/ImageMagick/ImageMagick)
*   **Video Conversion:** [FFmpeg](https://github.com/FFmpeg/FFmpeg)
*   **Exif Support:** [ExifTool](https://github.com/exiftool/exiftool)
*   **Face Detection:** [face\_recognition](https://github.com/ageitgey/face_recognition)
*   **Face Classification/Clusterization:** [scikit-learn](https://scikit-learn.org/) and [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
*   **Image Captioning:** [im2txt](https://github.com/HughKu/Im2txt)
*   **Scene Classification:** [places365](http://places.csail.mit.edu/)
*   **Reverse Geocoding:** [Mapbox](https://www.mapbox.com/) (API key required; free for the first 50,000 lookups per month)

[discord]: https://discord.gg/xwRvtSDGWb