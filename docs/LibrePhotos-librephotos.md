# LibrePhotos: Your Self-Hosted Photo Management Solution

Easily organize, manage, and share your photo and video library with LibrePhotos, a powerful and open-source alternative to cloud-based photo services.  [Visit the original repository on GitHub](https://github.com/LibrePhotos/librephotos).

[![Discord](https://img.shields.io/discord/784619049208250388?style=plastic)][discord] [![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&style=plastic&up_color=blue&up_message=online&url=https%3A%2F%2Flibrephotos.com)](https://librephotos.com/)
[![Read the docs](https://img.shields.io/static/v1?label=Read&message=the%20docs&color=blue&style=plastic)](https://docs.librephotos.com/) [![GitHub contributors](https://img.shields.io/github/contributors/librephotos/librephotos?style=plastic)](https://github.com/LibrePhotos/librephotos/graphs/contributors)
<a href="https://hosted.weblate.org/engage/librephotos/">
<img src="https://hosted.weblate.org/widgets/librephotos/-/librephotos-frontend/svg-badge.svg" alt="Translation status" />
</a>

## Key Features of LibrePhotos

*   **Comprehensive Media Support:** Handles photos (including RAW files) and videos.
*   **Organized Viewing:** Timeline view for easy browsing and event-based album generation ("Thursday in Berlin").
*   **Automated Organization:**  Scans your file system for photos, and offers face recognition, object/scene detection, and semantic image search.
*   **Multi-User Support:**  Share your library with friends and family.
*   **Advanced Features:** Includes reverse geocoding for location-based organization, and search by metadata.

## Demos

*   **Stable Demo:** https://demo1.librephotos.com/ (user: `demo`, password: `demo1234`)
*   **Development Demo:** https://demo2.librephotos.com/ (same credentials)

## Installation

Find detailed installation instructions in our comprehensive [documentation](https://docs.librephotos.com/docs/installation/standard-install).

## How to Contribute & Help Out

We welcome your contributions! Here's how you can help make LibrePhotos even better:

*   ⭐ **Star the Repository:** Show your support!
*   🚀 **Development:** Get started in under 30 minutes with our [development guide](https://docs.librephotos.com/docs/development/dev-install).
*   🗒️ **Documentation:** Improve our documentation via pull requests [here](https://github.com/LibrePhotos/librephotos.docs).
*   🧪 **Testing:** Test the `dev` tag and report any bugs you find.
*   🧑‍🤝‍🧑 **Outreach:** Spread the word about LibrePhotos!
*   🌐 **Translations:** Make LibrePhotos accessible worldwide through [weblate](https://hosted.weblate.org/engage/librephotos/).
*   💸 [**Donate**](https://github.com/sponsors/derneuere) to support the developers.

## Technology Stack

LibrePhotos leverages several powerful open-source tools:

*   **Image Conversion:** [ImageMagick](https://github.com/ImageMagick/ImageMagick)
*   **Video Conversion:** [FFmpeg](https://github.com/FFmpeg/FFmpeg)
*   **Exif Support:** [ExifTool](https://github.com/exiftool/exiftool)
*   **Face Detection:** [face_recognition](https://github.com/ageitgey/face_recognition)
*   **Face Classification:** [scikit-learn](https://scikit-learn.org/) and [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
*   **Image Captioning:** [im2txt](https://github.com/HughKu/Im2txt)
*   **Scene Classification:** [places365](http://places.csail.mit.edu/)
*   **Reverse Geocoding:** [Mapbox](https://www.mapbox.com/) (requires an API key; first 50,000 lookups/month are free)

[discord]: https://discord.gg/xwRvtSDGWb