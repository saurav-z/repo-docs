# LibrePhotos: Your Self-Hosted Photo Management Solution

Tired of cloud photo services? **LibrePhotos** is a powerful, open-source photo management solution that gives you complete control over your memories. (Check out the original repo at: [https://github.com/LibrePhotos/librephotos](https://github.com/LibrePhotos/librephotos))

[![Discord](https://img.shields.io/discord/784619049208250388?style=plastic)][discord] [![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&style=plastic&up_color=blue&up_message=online&url=https%3A%2F%2Flibrephotos.com)](https://librephotos.com/)
[![Read the docs](https://img.shields.io/static/v1?label=Read&message=the%20docs&color=blue&style=plastic)](https://docs.librephotos.com/) [![GitHub contributors](https://img.shields.io/github/contributors/librephotos/librephotos?style=plastic)](https://github.com/LibrePhotos/librephotos/graphs/contributors)
<a href="https://hosted.weblate.org/engage/librephotos/">
<img src="https://hosted.weblate.org/widgets/librephotos/-/librephotos-frontend/svg-badge.svg" alt="Translation status" />
</a>

![](https://github.com/LibrePhotos/librephotos/blob/dev/screenshots/mockups_main_fhd.png?raw=true)
<sub>Mockup designed by rawpixel.com / Freepik</sub>

## Key Features

*   **Comprehensive Media Support:** Handles all photo types, including RAW files, and videos.
*   **Intelligent Organization:** Timeline view, event-based album generation ("Thursday in Berlin").
*   **Advanced Search:** Semantic image search, search by metadata, and object/scene detection for easy retrieval.
*   **Powerful AI Capabilities:** Face recognition, face classification, and image captioning.
*   **Multi-User Support:** Share your photos and memories with others.
*   **Reverse Geocoding:** Automatically tag photos with location data.

## Demos

*   **Stable Demo:** [https://demo1.librephotos.com/](https://demo1.librephotos.com/) - User: `demo`, Password: `demo1234` (with sample images).
*   **Development Demo:** [https://demo2.librephotos.com/](https://demo2.librephotos.com/) - User: `demo`, Password: `demo1234`.

## Getting Started

*   **Installation:** Follow the step-by-step instructions in our detailed [documentation](https://docs.librephotos.com/docs/installation/standard-install).

## How to Contribute

We welcome contributions! Here's how you can help:

*   ⭐ **Star** the repository to show your support!
*   🚀 **Develop:** Get started quickly with [this guide](https://docs.librephotos.com/docs/development/dev-install).
*   🗒️ **Documentation:** Improve the documentation by submitting a pull request [here](https://github.com/LibrePhotos/librephotos.docs).
*   🧪 **Testing:** Help find bugs by using the `dev` tag and reporting any issues.
*   🧑‍🤝‍🧑 **Outreach:** Spread the word about LibrePhotos!
*   🌐 **Translations:** Help make LibrePhotos accessible to more people through [Weblate](https://hosted.weblate.org/engage/librephotos/).
*   💸 **Donate:** Support the developers [here](https://github.com/sponsors/derneuere).

## Technologies Used

LibrePhotos leverages several open-source technologies:

*   **Image Conversion:** [ImageMagick](https://github.com/ImageMagick/ImageMagick)
*   **Video Conversion:** [FFmpeg](https://github.com/FFmpeg/FFmpeg)
*   **Exif Support:** [ExifTool](https://github.com/exiftool/exiftool)
*   **Face Detection:** [face_recognition](https://github.com/ageitgey/face_recognition)
*   **Face Classification/Clustering:** [scikit-learn](https://scikit-learn.org/) and [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
*   **Image Captioning:** [im2txt](https://github.com/HughKu/Im2txt)
*   **Scene Classification:** [places365](http://places.csail.mit.edu/)
*   **Reverse Geocoding:** [Mapbox](https://www.mapbox.com/) (API key required; free tier available).

[discord]: https://discord.gg/xwRvtSDGWb