# LibrePhotos: Your Self-Hosted, Open-Source Photo Management Solution

**Tired of relying on cloud services for your precious memories?** LibrePhotos is a powerful, open-source photo management system that puts you in control of your photos and videos, offering robust features and complete privacy. Explore the original project on GitHub: [https://github.com/LibrePhotos/librephotos](https://github.com/LibrePhotos/librephotos).

[![Discord](https://img.shields.io/discord/784619049208250388?style=plastic)][discord]
[![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&style=plastic&up_color=blue&up_message=online&url=https%3A%2F%2Flibrephotos.com)](https://librephotos.com/)
[![Read the docs](https://img.shields.io/static/v1?label=Read&message=the%20docs&color=blue&style=plastic)](https://docs.librephotos.com/)
[![GitHub contributors](https://img.shields.io/github/contributors/librephotos/librephotos?style=plastic)](https://github.com/LibrePhotos/librephotos/graphs/contributors)
<a href="https://hosted.weblate.org/engage/librephotos/">
<img src="https://hosted.weblate.org/widgets/librephotos/-/librephotos-frontend/svg-badge.svg" alt="Translation status" />
</a>

## Key Features

*   **Comprehensive Media Support:** Handles photos of all types, including RAW files, and videos.
*   **Organized Viewing:** Timeline view for easy browsing and a clear chronological overview of your photos.
*   **Automated Organization:** Scans your file system, creating albums and tagging photos, including event-based albums.
*   **Advanced Search:** Powerful semantic image search, search by metadata, and object/scene detection to find photos quickly.
*   **Face Recognition and Clustering:** Automatic face recognition and classification to organize people in your photos.
*   **Geotagging & Location Services:** Reverse geocoding to automatically add location data to your photos.
*   **Multiuser Support:** Share your photos with others while maintaining individual privacy controls.

## Demos
*   **Stable Demo:** [https://demo1.librephotos.com/](https://demo1.librephotos.com/) - User: `demo`, Password: `demo1234`
*   **Development Demo:** [https://demo2.librephotos.com/](https://demo2.librephotos.com/) - User: `demo`, Password: `demo1234`

## Installation

Detailed installation instructions are available in the [LibrePhotos documentation](https://docs.librephotos.com/docs/installation/standard-install).

## How to Contribute

LibrePhotos thrives on community contributions! Here's how you can help:

*   ⭐ **Star** the repository on GitHub!
*   🚀 **Developing:** Get started in under 30 minutes using [this guide](https://docs.librephotos.com/docs/development/dev-install).
*   🗒️ **Documentation:** Improve the documentation by submitting pull requests [here](https://github.com/LibrePhotos/librephotos.docs).
*   🧪 **Testing:** Help find bugs by using the `dev` tag and reporting issues.
*   🧑‍🤝‍🧑 **Outreach:** Spread the word about LibrePhotos.
*   🌐 **Translations:** Make LibrePhotos accessible in more languages using [Weblate](https://hosted.weblate.org/engage/librephotos/).
*   💸 [**Donate**](https://github.com/sponsors/derneuere) to support the developers.

## Technologies Used

LibrePhotos leverages several open-source technologies:

*   **Image Conversion:** [ImageMagick](https://github.com/ImageMagick/ImageMagick)
*   **Video Conversion:** [FFmpeg](https://github.com/FFmpeg/FFmpeg)
*   **Exif Support:** [ExifTool](https://github.com/exiftool/exiftool)
*   **Face Detection:** [face_recognition](https://github.com/ageitgey/face_recognition)
*   **Face Classification:** [scikit-learn](https://scikit-learn.org/) and [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
*   **Image Captioning:** [im2txt](https://github.com/HughKu/Im2txt)
*   **Scene Classification:** [places365](http://places.csail.mit.edu/)
*   **Reverse Geocoding:** [Mapbox](https://www.mapbox.com/) (requires an API key)

[discord]: https://discord.gg/xwRvtSDGWb