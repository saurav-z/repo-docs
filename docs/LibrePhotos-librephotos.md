[![Discord](https://img.shields.io/discord/784619049208250388?style=plastic)][discord] [![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&style=plastic&up_color=blue&up_message=online&url=https%3A%2F%2Flibrephotos.com)](https://librephotos.com/)
[![Read the docs](https://img.shields.io/static/v1?label=Read&message=the%20docs&color=blue&style=plastic)](https://docs.librephotos.com/) [![GitHub contributors](https://img.shields.io/github/contributors/librephotos/librephotos?style=plastic)](https://github.com/LibrePhotos/librephotos/graphs/contributors)
<a href="https://hosted.weblate.org/engage/librephotos/">
<img src="https://hosted.weblate.org/widgets/librephotos/-/librephotos-frontend/svg-badge.svg" alt="Translation status" />
</a>

# LibrePhotos: Your Self-Hosted, Open-Source Photo Management Solution

Tired of relying on proprietary photo storage? **LibrePhotos** is a powerful and open-source photo management solution, giving you complete control over your memories.  [Check out the original repo](https://github.com/LibrePhotos/librephotos)!

![LibrePhotos Screenshot](https://github.com/LibrePhotos/librephotos/blob/dev/screenshots/mockups_main_fhd.png?raw=true)
<sub>Mockup designed by rawpixel.com / Freepik</sub>

## Key Features

*   **Comprehensive Media Support:** Upload, view, and manage all your photos and videos, including RAW files.
*   **Organized Timeline View:** Browse your memories chronologically with an intuitive timeline.
*   **Automated Organization:** Automatically scans your file system and creates albums based on events and locations.
*   **Advanced Search Capabilities:** Find photos easily using face recognition, object detection, semantic image search, and metadata.
*   **Multi-User Support:** Share and collaborate with others.
*   **Powerful Face Recognition:** Effortlessly identify and tag people in your photos.
*   **Location-Based Features:** Reverse geocoding to see where your photos were taken.
*   **Scene and Object Detection:** Automatic scene and object recognition within your images.
*   **Built-in Video Support:** Play and manage all your videos.

## Installation & Getting Started

Detailed installation instructions are available in our comprehensive [documentation](https://docs.librephotos.com/docs/installation/standard-install).

*   **Stable Demo:** Explore a live demo with sample images at: [https://demo1.librephotos.com/](https://demo1.librephotos.com/). Use `demo` as the username and `demo1234` as the password.
*   **Development Demo:** See the latest features in action at: [https://demo2.librephotos.com/](https://demo2.librephotos.com/) (same credentials).

## How to Contribute & Help Out

We welcome contributions of all kinds!

*   ⭐ **Star the Repository:** Show your support and help others discover LibrePhotos.
*   🚀 **Develop:** Get started with development quickly by following [this guide](https://docs.librephotos.com/docs/development/dev-install).
*   🗒️ **Document:** Improve the documentation by submitting pull requests [here](https://github.com/LibrePhotos/librephotos.docs).
*   🧪 **Test:** Help find bugs by using the `dev` tag and reporting any issues.
*   🧑‍🤝‍🧑 **Spread the Word:** Share LibrePhotos with others and help them get started.
*   🌐 **Translate:** Make LibrePhotos accessible to a wider audience through [weblate](https://hosted.weblate.org/engage/librephotos/).
*   💸 **Donate:** Support the developers via [GitHub Sponsors](https://github.com/sponsors/derneuere).

## Technologies Used

LibrePhotos leverages several powerful technologies:

*   **Image Conversion:** [ImageMagick](https://github.com/ImageMagick/ImageMagick)
*   **Video Conversion:** [FFmpeg](https://github.com/FFmpeg/FFmpeg)
*   **Exif Support:** [ExifTool](https://github.com/exiftool/exiftool)
*   **Face Detection:** [face_recognition](https://github.com/ageitgey/face_recognition)
*   **Face Classification/Clustering:** [scikit-learn](https://scikit-learn.org/) and [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
*   **Image Captioning:** [im2txt](https://github.com/HughKu/Im2txt)
*   **Scene Classification:** [places365](http://places.csail.mit.edu/)
*   **Reverse Geocoding:** [Mapbox](https://www.mapbox.com/) (Requires API key; first 50,000 geocode lookups are free monthly)

[discord]: https://discord.gg/xwRvtSDGWb