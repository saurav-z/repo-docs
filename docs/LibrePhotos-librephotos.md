[![Discord](https://img.shields.io/discord/784619049208250388?style=plastic)][discord]
[![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&style=plastic&up_color=blue&up_message=online&url=https%3A%2F%2Flibrephotos.com)](https://librephotos.com/)
[![Read the docs](https://img.shields.io/static/v1?label=Read&message=the%20docs&color=blue&style=plastic)](https://docs.librephotos.com/)
[![GitHub contributors](https://img.shields.io/github/contributors/librephotos/librephotos?style=plastic)](https://github.com/LibrePhotos/librephotos/graphs/contributors)
<a href="https://hosted.weblate.org/engage/librephotos/">
<img src="https://hosted.weblate.org/widgets/librephotos/-/librephotos-frontend/svg-badge.svg" alt="Translation status" />
</a>

# LibrePhotos: Your Self-Hosted, Open-Source Photo Management Solution

LibrePhotos is a powerful and feature-rich photo management application that allows you to take control of your photos with a self-hosted, open-source solution.  ([See the original repository](https://github.com/LibrePhotos/librephotos))

## Key Features

*   📸 **Comprehensive Media Support:** Supports photos (including RAW) and videos.
*   📅 **Organized Viewing:** Timeline view for easy browsing.
*   📁 **Automatic Scanning:** Scans your file system to find and organize photos.
*   👥 **Multi-User Support:** Allows multiple users to access and manage photos.
*   🎉 **Smart Album Generation:** Creates albums based on events and locations.
*   👤 **Advanced AI Features:** Includes face recognition, classification, and semantic image search.
*   🗺️ **Geotagging & Search:** Reverse geocoding for location-based organization and search by metadata.
*   🔍 **Object & Scene Detection:** Automatically identifies objects and scenes within your photos for improved search.

## Get Started

*   **Demo:** Explore a live demo of LibrePhotos:
    *   [Stable Demo](https://demo1.librephotos.com/) - user: `demo`, password: `demo1234`
    *   [Development Demo](https://demo2.librephotos.com/) - user: `demo`, password: `demo1234`
*   **Installation:** Follow the step-by-step instructions in the [documentation](https://docs.librephotos.com/docs/installation/standard-install).
*   **Development Videos:** Watch development videos on [Niaz Faridani-Rad's channel](https://www.youtube.com/channel/UCZJ2pk2BPKxwbuCV9LWDR0w)
*   **Community:** Join the LibrePhotos community on [Discord][discord]

## How to Contribute

We welcome contributions from the community! Here's how you can help:

*   ⭐ **Star** the repository on GitHub!
*   🚀 **Develop:** Get started by following the [development guide](https://docs.librephotos.com/docs/development/dev-install).
*   🗒️ **Documentation:** Improve the documentation by submitting pull requests [here](https://github.com/LibrePhotos/librephotos.docs).
*   🧪 **Testing:** Help find bugs by testing the `dev` branch and reporting any issues.
*   🧑‍🤝‍🧑 **Outreach:** Share LibrePhotos with others and help them get started.
*   🌐 **Translations:** Help translate LibrePhotos with [weblate](https://hosted.weblate.org/engage/librephotos/).
*   💸 **Donate:** Support the developers of LibrePhotos via [GitHub Sponsors](https://github.com/sponsors/derneuere).

## Technologies Used

LibrePhotos leverages several powerful technologies:

*   **Image Conversion:** [ImageMagick](https://github.com/ImageMagick/ImageMagick)
*   **Video Conversion:** [FFmpeg](https://github.com/FFmpeg/FFmpeg)
*   **Exif Support:** [ExifTool](https://github.com/exiftool/exiftool)
*   **Face Detection:** [face_recognition](https://github.com/ageitgey/face_recognition)
*   **Face Classification/Clusterization:** [scikit-learn](https://scikit-learn.org/) and [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
*   **Image Captioning:** [im2txt](https://github.com/HughKu/Im2txt)
*   **Scene Classification:** [places365](http://places.csail.mit.edu/)
*   **Reverse Geocoding:** [Mapbox](https://www.mapbox.com/) (requires an API key; first 50,000 geocode lookups are free every month)

[discord]: https://discord.gg/xwRvtSDGWb