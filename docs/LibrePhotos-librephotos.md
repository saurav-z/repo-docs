# LibrePhotos: Your Open-Source, Self-Hosted Photo Management Solution

Tired of relying on big tech for your photos? **LibrePhotos is a powerful and open-source photo management solution that puts you in control of your memories.**  This project provides a comprehensive platform for organizing, viewing, and sharing your photos and videos.

[![Discord](https://img.shields.io/discord/784619049208250388?style=plastic)][discord] [![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&style=plastic&up_color=blue&up_message=online&url=https%3A%2F%2Flibrephotos.com)](https://librephotos.com/)
[![Read the docs](https://img.shields.io/static/v1?label=Read&message=the%20docs&color=blue&style=plastic)](https://docs.librephotos.com/) [![GitHub contributors](https://img.shields.io/github/contributors/librephotos/librephotos?style=plastic)](https://github.com/LibrePhotos/librephotos/graphs/contributors)
<a href="https://hosted.weblate.org/engage/librephotos/">
<img src="https://hosted.weblate.org/widgets/librephotos/-/librephotos-frontend/svg-badge.svg" alt="Translation status" />
</a>

## Key Features

*   **Comprehensive Media Support:** Upload, organize, and view all your photos and videos, including RAW image formats.
*   **Intuitive Timeline View:** Easily browse your memories chronologically.
*   **Intelligent Organization:**  Generate albums automatically based on events and locations ("Thursday in Berlin").
*   **Advanced Search Capabilities:** Find photos using facial recognition, object/scene detection, metadata, and semantic image search.
*   **Multi-User Support:** Share your photo library with family and friends.
*   **Face Detection & Recognition:** Automated face detection and classification to organize photos by people.
*   **Reverse Geocoding:** Automatically add location data to your photos.
*   **Object & Scene Detection:** AI-powered tagging to help you find specific content.

## Getting Started

*   **Demo:**  Explore a live demo!
    *   Stable: [https://demo1.librephotos.com/](https://demo1.librephotos.com/) (user: `demo`, password: `demo1234`)
    *   Development: [https://demo2.librephotos.com/](https://demo2.librephotos.com/) (user: `demo`, password: `demo1234`)
*   **Installation:**  Follow the step-by-step installation guide in our [documentation](https://docs.librephotos.com/docs/installation/standard-install).
*   **Development Videos:** Watch development videos on [Niaz Faridani-Rad's channel](https://www.youtube.com/channel/UCZJ2pk2BPKxwbuCV9LWDR0w)
*   **Join the Community:** Connect with other users and developers on [Discord][discord].

## Contributing to LibrePhotos

We welcome contributions from the community!

*   ⭐ **Star** the repository to show your support!
*   🚀 **Develop:** Get started with a development environment in under 30 minutes by following [this guide](https://docs.librephotos.com/docs/development/dev-install).
*   🗒️ **Document:** Improve the documentation by submitting pull requests [here](https://github.com/LibrePhotos/librephotos.docs).
*   🧪 **Test:** Help find bugs by using the `dev` tag and reporting issues.
*   🧑‍🤝‍🧑 **Outreach:** Spread the word about LibrePhotos!
*   🌐 **Translate:** Make LibrePhotos accessible to more people with [weblate](https://hosted.weblate.org/engage/librephotos/).
*   💸 **Donate:** Support the developers of LibrePhotos through [GitHub Sponsors](https://github.com/sponsors/derneuere).

## Technologies Used

LibrePhotos leverages several open-source technologies:

*   **Image Conversion:** [ImageMagick](https://github.com/ImageMagick/ImageMagick)
*   **Video Conversion:** [FFmpeg](https://github.com/FFmpeg/FFmpeg)
*   **Exif Support:** [ExifTool](https://github.com/exiftool/exiftool)
*   **Face Detection:** [face\_recognition](https://github.com/ageitgey/face_recognition)
*   **Face Classification/Clustering:** [scikit-learn](https://scikit-learn.org/) and [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
*   **Image Captioning:** [im2txt](https://github.com/HughKu/Im2txt)
*   **Scene Classification:** [places365](http://places.csail.mit.edu/)
*   **Reverse Geocoding:** [Mapbox](https://www.mapbox.com/) (API key required - first 50,000 lookups free per month)

## Learn More

Explore the source code and join the community on [GitHub](https://github.com/LibrePhotos/librephotos).

[discord]: https://discord.gg/xwRvtSDGWb