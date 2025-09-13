# Buzz: Offline Audio Transcription and Translation Powered by Whisper

**Effortlessly transcribe and translate audio files on your computer with Buzz, a powerful application built on OpenAI's Whisper technology.** [Get the original code here](https://github.com/chidiwilliams/buzz).

[![MIT License](https://img.shields.io/badge/license-MIT-green)](https://github.com/chidiwilliams/buzz)
[![CI](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml/badge.svg)](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/chidiwilliams/buzz/branch/main/graph/badge.svg?token=YJSB8S2VEP)](https://codecov.io/github/chidiwilliams/buzz)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/chidiwilliams/buzz)
[![Github all releases](https://img.shields.io/github/downloads/chidiwilliams/buzz/total.svg)](https://GitHub.com/chidiwilliams/buzz/releases/)

## Key Features:

*   **Offline Transcription:** Transcribe audio files without an internet connection.
*   **Multi-Language Support:** Powered by Whisper to transcribe and translate many languages.
*   **Cross-Platform:** Available for macOS, Windows, and Linux.
*   **Easy Installation:** Install via PyPI, brew, winget, Flatpak, or Snap.
*   **GPU Acceleration (Optional):**  Utilize your NVIDIA GPU for faster transcription (PyPI installation).
*   **Mac App Store Version:** Experience a native macOS version with enhanced features.

## Why Choose Buzz?

Buzz offers a convenient and private solution for audio transcription and translation. With its offline capabilities, you can process audio files anywhere, without worrying about data privacy or internet connectivity.

## Installation

Choose your operating system below:

### macOS

*   **Homebrew:** `brew install --cask buzz`
*   **Download:** Download the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   **Winget:** `winget install ChidiWilliams.Buzz`
*   **Download:** Download and run the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).
    *   *Note: App is not signed; you may get a warning.  Select "More info" -> "Run anyway."*
*   **GPU Support (PyPI):**  Follow the instructions in the original README to enable GPU acceleration.

### Linux

*   **Flatpak:**
    ```shell
    flatpak install flathub io.github.chidiwilliams.Buzz
    ```
*   **Snap:**
    ```shell
    sudo apt-get install libportaudio2 libcanberra-gtk-module libcanberra-gtk3-module
    sudo snap install buzz
    sudo snap connect buzz:password-manager-service
    ```

### PyPI

1.  Install [ffmpeg](https://www.ffmpeg.org/download.html)
2.  Install Buzz:
    ```shell
    pip install buzz-captions
    python -m buzz
    ```
    *   **GPU Support:** (follow original README for installation)

## Screenshots

[Include the existing screenshots here]

## Additional Resources
*   [Documentation](https://chidiwilliams.github.io/buzz/)
*   [Buzz Captions on the Mac App Store](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200)
*   [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version)
```

Key improvements and SEO optimizations:

*   **Clear Headline:** Includes the primary keywords: "Audio Transcription," "Translation," and "Whisper."
*   **One-Sentence Hook:** Catches attention immediately.
*   **Keyword Optimization:**  Uses relevant keywords throughout (e.g., "offline," "transcribe," "translate," "cross-platform").
*   **Bulleted Key Features:** Highlights the main benefits, making the information easy to scan.
*   **Installation Instructions:** Clearly separates installation instructions by operating system, with links to important tools.
*   **Call to Action:**  Suggests to download the Mac App Store version.
*   **Links to Resources:**  Provides direct links to documentation and the app store.
*   **Concise Language:**  Removes unnecessary words to improve readability.
*   **Clean Formatting:**  Uses Markdown effectively for better presentation.