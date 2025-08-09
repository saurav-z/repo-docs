[![Discord](https://img.shields.io/discord/784619049208250388?style=plastic)][discord] [![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&style=plastic&up_color=blue&up_message=online&url=https%3A%2F%2Flibrephotos.com)](https://librephotos.com/)
[![Read the docs](https://img.shields.io/static/v1?label=Read&message=the%20docs&color=blue&style=plastic)](https://docs.librephotos.com/) [![GitHub contributors](https://img.shields.io/github/contributors/librephotos/librephotos?style=plastic)](https://github.com/LibrePhotos/librephotos/graphs/contributors)
<a href="https://hosted.weblate.org/engage/librephotos/">
<img src="https://hosted.weblate.org/widgets/librephotos/-/librephotos-frontend/svg-badge.svg" alt="Translation status" />
</a>

# LibrePhotos: Your Self-Hosted, Open-Source Photo Management Solution

**Tired of relying on cloud services for your photos?** LibrePhotos provides a powerful, self-hosted photo management solution with advanced features and complete control over your data.  [View the original repo on GitHub](https://github.com/LibrePhotos/librephotos).

![LibrePhotos Screenshot](https://github.com/LibrePhotos/librephotos/blob/dev/screenshots/mockups_main_fhd.png?raw=true)
<sub>Mockup designed by rawpixel.com / Freepik</sub>

## Key Features of LibrePhotos

*   **Comprehensive Media Support:** Supports photos of all types (including RAW) and videos.
*   **Intelligent Organization:**  Timeline view, automatic album generation based on events, and metadata-driven search.
*   **Advanced Image Analysis:** Includes face recognition and classification, object/scene detection, and reverse geocoding.
*   **Multi-User Support:**  Share and manage your photos with others.
*   **Powerful Search:** Semantic image search and search by metadata make finding your photos easy.
*   **Easy to Explore:**  Available demo with sample images: [Stable Demo](https://demo1.librephotos.com/) and [Development Demo](https://demo2.librephotos.com/) (user: demo, password: demo1234).

## Getting Started

### Installation

Detailed installation instructions are available in the [LibrePhotos Documentation](https://docs.librephotos.com/docs/installation/standard-install).

### How to Contribute

We welcome contributions!  Here's how you can help:

*   ⭐ **Star** the repository on GitHub to show your support!
*   🚀 **Develop:**  Get started developing in under 30 minutes by following [this guide](https://docs.librephotos.com/docs/development/dev-install).
*   🗒️ **Document:** Improve the documentation via pull requests [here](https://github.com/LibrePhotos/librephotos.docs).
*   🧪 **Test:** Help find bugs by using the ```dev``` tag and reporting issues.
*   🧑‍🤝‍🧑 **Outreach:** Spread the word about LibrePhotos and help others get started.
*   🌐 **Translate:** Make LibrePhotos accessible to more people through [Weblate](https://hosted.weblate.org/engage/librephotos/).
*   💸 **Donate:** Support the developers via [GitHub Sponsors](https://github.com/sponsors/derneuere).

## Technologies Used

LibrePhotos leverages the following technologies:

*   **Image Conversion:** [ImageMagick](https://github.com/ImageMagick/ImageMagick)
*   **Video Conversion:** [FFmpeg](https://github.com/FFmpeg/FFmpeg)
*   **Exif Support:** [ExifTool](https://github.com/exiftool/exiftool)
*   **Face Detection:** [face\_recognition](https://github.com/ageitgey/face_recognition)
*   **Face Classification/Clustering:** [scikit-learn](https://scikit-learn.org/) and [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
*   **Image Captioning:** [im2txt](https://github.com/HughKu/Im2txt)
*   **Scene Classification:** [places365](http://places.csail.mit.edu/)
*   **Reverse Geocoding:** [Mapbox](https://www.mapbox.com/) (API key required - first 50,000 lookups free per month)

[discord]: https://discord.gg/xwRvtSDGWb