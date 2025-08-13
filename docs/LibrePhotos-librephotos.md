[![Discord](https://img.shields.io/discord/784619049208250388?style=plastic)][discord] [![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&style=plastic&up_color=blue&up_message=online&url=https%3A%2F%2Flibrephotos.com)](https://librephotos.com/)
[![Read the docs](https://img.shields.io/static/v1?label=Read&message=the%20docs&color=blue&style=plastic)](https://docs.librephotos.com/) [![GitHub contributors](https://img.shields.io/github/contributors/librephotos/librephotos?style=plastic)](https://github.com/LibrePhotos/librephotos/graphs/contributors)
<a href="https://hosted.weblate.org/engage/librephotos/">
<img src="https://hosted.weblate.org/widgets/librephotos/-/librephotos-frontend/svg-badge.svg" alt="Translation status" />
</a>

# LibrePhotos: Your Self-Hosted, Open-Source Photo Management Solution

**Tired of restrictive cloud storage and want complete control over your photo library?** LibrePhotos offers a powerful, open-source, and self-hosted solution for organizing, managing, and sharing your photos and videos. [Explore the LibrePhotos project on GitHub](https://github.com/LibrePhotos/librephotos)!

<img src="https://github.com/LibrePhotos/librephotos/blob/dev/screenshots/mockups_main_fhd.png?raw=true" alt="LibrePhotos Interface" width="800"/>
<sub>Mockup designed by rawpixel.com / Freepik</sub>

## Key Features

*   🖼️ **Comprehensive Media Support:** Supports photos of all types, including RAW images, and videos.
*   🗓️ **Intelligent Organization:** Offers a timeline view and automatically generates albums based on events (e.g., "Thursday in Berlin").
*   👤 **Advanced Search & Discovery:** Enables face recognition, object/scene detection, semantic image search, and metadata-based search for effortless finding.
*   🧑‍🤝‍🧑 **Multi-User Support:** Share your photos with others while maintaining control.
*   🌍 **Geotagging & Mapping:** Utilizes reverse geocoding to associate your photos with locations.
*   💻 **Self-Hosted & Open Source:** Take control of your data and host your photos on your own server.

## Getting Started

*   **Live Demos:**
    *   **Stable Demo:** [https://demo1.librephotos.com/](https://demo1.librephotos.com/) (User: `demo`, Password: `demo1234`) - with sample images.
    *   **Development Demo:** [https://demo2.librephotos.com/](https://demo2.librephotos.com/) (User: `demo`, Password: `demo1234`)
*   **Installation:** Detailed installation instructions are available in the [LibrePhotos documentation](https://docs.librephotos.com/docs/installation/standard-install).
*   **Development Videos:** Watch development progress on [Niaz Faridani-Rad's channel](https://www.youtube.com/channel/UCZJ2pk2BPKxwbuCV9LWDR0w)
*   **Join the Community:** Connect with other users and developers on our [Discord server][discord].

## How to Contribute

*   ⭐ **Star the Repository:** Show your support by starring this project!
*   🚀 **Development:** Contribute to the project, get started quickly with this [guide](https://docs.librephotos.com/docs/development/dev-install).
*   🗒️ **Documentation:** Improve documentation via pull requests [here](https://github.com/LibrePhotos/librephotos.docs).
*   🧪 **Testing:** Help identify bugs by testing with the `dev` tag and reporting any issues.
*   🧑‍🤝‍🧑 **Outreach:** Share LibrePhotos with others and help them get started.
*   🌐 **Translations:** Make LibrePhotos available in more languages through [weblate](https://hosted.weblate.org/engage/librephotos/).
*   💸 **Donate:** Support the developers of LibrePhotos via [GitHub Sponsors](https://github.com/sponsors/derneuere).

## Technologies Used

*   **Image Conversion:** [ImageMagick](https://github.com/ImageMagick/ImageMagick)
*   **Video Conversion:** [FFmpeg](https://github.com/FFmpeg/FFmpeg)
*   **Exif Support:** [ExifTool](https://github.com/exiftool/exiftool)
*   **Face Detection:** [face_recognition](https://github.com/ageitgey/face_recognition)
*   **Face Classification/Clustering:** [scikit-learn](https://scikit-learn.org/) and [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
*   **Image Captioning:** [im2txt](https://github.com/HughKu/Im2txt)
*   **Scene Classification:** [places365](http://places.csail.mit.edu/)
*   **Reverse Geocoding:** [Mapbox](https://www.mapbox.com/) (API key required; free for the first 50,000 geocode lookups/month)

[discord]: https://discord.gg/xwRvtSDGWb