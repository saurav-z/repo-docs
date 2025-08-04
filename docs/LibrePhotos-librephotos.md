# LibrePhotos: Your Self-Hosted Photo Management Solution

Tired of cloud-based photo storage? **LibrePhotos** offers a powerful, open-source alternative for managing your photos and videos, giving you complete control over your media. ([Original Repository](https://github.com/LibrePhotos/librephotos))

[![Discord](https://img.shields.io/discord/784619049208250388?style=plastic)][discord]
[![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&style=plastic&up_color=blue&up_message=online&url=https%3A%2F%2Flibrephotos.com)](https://librephotos.com/)
[![Read the docs](https://img.shields.io/static/v1?label=Read&message=the%20docs&color=blue&style=plastic)](https://docs.librephotos.com/)
[![GitHub contributors](https://img.shields.io/github/contributors/librephotos/librephotos?style=plastic)](https://github.com/LibrePhotos/librephotos/graphs/contributors)
<a href="https://hosted.weblate.org/engage/librephotos/">
<img src="https://hosted.weblate.org/widgets/librephotos/-/librephotos-frontend/svg-badge.svg" alt="Translation status" />
</a>

## Key Features

*   ✅ **Comprehensive Media Support:** Handles all photo types, including RAW files, and videos.
*   📅 **Intuitive Timeline View:** Browse your photos chronologically.
*   📁 **File System Scanning:** Easily import photos from your existing file structure.
*   👥 **Multi-User Support:** Share your photos and manage access for multiple users.
*   🎉 **Intelligent Album Creation:** Automatically generates albums based on events and locations (e.g., "Thursday in Berlin").
*   😀 **Advanced AI Features:**
    *   Face recognition and classification
    *   Object and scene detection
    *   Semantic image search
*   🗺️ **Location-Aware:** Reverse geocoding for location-based photo organization.
*   🔎 **Metadata Search:** Search by keywords and other metadata.

## Demo

*   **Stable Demo:** [https://demo1.librephotos.com/](https://demo1.librephotos.com/) (User: `demo`, Password: `demo1234`)
*   **Development Demo:** [https://demo2.librephotos.com/](https://demo2.librephotos.com/) (User: `demo`, Password: `demo1234`)

## Installation

Detailed installation instructions are available in the [documentation](https://docs.librephotos.com/docs/installation/standard-install).

## How to Contribute

We welcome contributions! Here's how you can help:

*   ⭐ **Star** the repository to show your support.
*   🚀 **Develop:** Get started quickly using [this guide](https://docs.librephotos.com/docs/development/dev-install).
*   🗒️ **Document:** Improve the documentation by submitting pull requests [here](https://github.com/LibrePhotos/librephotos.docs).
*   🧪 **Test:** Help find bugs by using the `dev` tag and reporting issues.
*   🧑‍🤝‍🧑 **Spread the Word:** Tell others about LibrePhotos.
*   🌐 **Translate:** Make LibrePhotos accessible in more languages via [weblate](https://hosted.weblate.org/engage/librephotos/).
*   💸 **Donate:** Support the developers via [GitHub Sponsors](https://github.com/sponsors/derneuere).

## Technologies Used

LibrePhotos leverages a range of powerful open-source technologies:

*   **Image Conversion:** [ImageMagick](https://github.com/ImageMagick/ImageMagick)
*   **Video Conversion:** [FFmpeg](https://github.com/FFmpeg/FFmpeg)
*   **Exif Support:** [ExifTool](https://github.com/exiftool/exiftool)
*   **Face Detection:** [face\_recognition](https://github.com/ageitgey/face_recognition)
*   **Face Classification:** [scikit-learn](https://scikit-learn.org/) and [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
*   **Image Captioning:** [im2txt](https://github.com/HughKu/Im2txt)
*   **Scene Classification:** [places365](http://places.csail.mit.edu/)
*   **Reverse Geocoding:** [Mapbox](https://www.mapbox.com/) (Requires an API key; first 50,000 lookups are free monthly.)

[discord]: https://discord.gg/xwRvtSDGWb