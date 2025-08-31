# Buzz: Offline Audio Transcription and Translation Powered by Whisper

**Effortlessly transcribe and translate audio files on your computer with Buzz, leveraging the power of OpenAI's Whisper.**  [Visit the original repository on GitHub](https://github.com/chidiwilliams/buzz).

[![MIT License](https://img.shields.io/badge/license-MIT-green)](https://github.com/chidiwilliams/buzz)
[![CI](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml/badge.svg)](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/chidiwilliams/buzz/branch/main/graph/badge.svg?token=YJSB8S2VEP)](https://codecov.io/github/chidiwilliams/buzz)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/chidiwilliams/buzz)
[![Github all releases](https://img.shields.io/github/downloads/chidiwilliams/buzz/total.svg)](https://GitHub.com/chidiwilliams/buzz/releases/)

![Buzz](./buzz/assets/buzz-banner.jpg)

## Key Features

*   **Offline Transcription & Translation:** Process audio files locally, ensuring privacy and speed.
*   **Powered by Whisper:** Utilizes the robust speech recognition capabilities of OpenAI's Whisper.
*   **Cross-Platform Compatibility:** Available for macOS, Windows, and Linux.
*   **Multiple Installation Options:** Install via PyPI, brew, winget, Flatpak, or Snap, offering flexibility.
*   **GPU Support:** Take advantage of NVIDIA GPU acceleration for faster processing (PyPI installation).
*   **Mac App Store Version:**  For an enhanced user experience, a Mac-native version is available with additional features like audio playback and transcript editing.

> Buzz is better on the App Store. Get a Mac-native version of Buzz with a cleaner look, audio playback, drag-and-drop import, transcript editing, search, and much more.
> <a href="https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&amp;itsct=apps_box_badge&amp;itscg=30200"><img src="https://toolbox.marketingtools.apple.com/api/badges/download-on-the-mac-app-store/black/en-us?size=250x83&amp;releaseDate=1679529600" alt="Download on the Mac App Store" /></a>

## Installation

Choose your preferred installation method:

### PyPI

1.  Install [ffmpeg](https://www.ffmpeg.org/download.html)
2.  Install Buzz: `pip install buzz-captions`
3.  Run Buzz: `python -m buzz`

    **GPU Support (Nvidia):** For PyPI installs on Windows with NVIDIA GPUs, ensure CUDA support for [torch](https://pytorch.org/get-started/locally/) and install additional packages, as detailed in the original README.

### macOS

*   **Homebrew:** `brew install --cask buzz`
*   **Direct Download:** Download the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   **Direct Download:** Download the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).  You may need to select `More info` -> `Run anyway` during installation because the app is not signed.
*   **Winget:** `winget install ChidiWilliams.Buzz`

### Linux

*   **Flatpak:** `flatpak install flathub io.github.chidiwilliams.Buzz`
*   **Snap:**
    ```bash
    sudo apt-get install libportaudio2 libcanberra-gtk-module libcanberra-gtk3-module
    sudo snap install buzz
    sudo snap connect buzz:password-manager-service
    ```

### Latest Development Version

For the latest features and bug fixes, see the [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version)

## Screenshots

<div style="display: flex; flex-wrap: wrap;">
    <img alt="File import" src="share/screenshots/buzz-1-import.png" style="max-width: 18%; margin-right: 1%;" />
    <img alt="Main screen" src="share/screenshots/buzz-2-main_screen.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Preferences" src="share/screenshots/buzz-3-preferences.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Model preferences" src="share/screenshots/buzz-3.2-model-preferences.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Transcript" src="share/screenshots/buzz-4-transcript.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Live recording" src="share/screenshots/buzz-5-live_recording.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Resize" src="share/screenshots/buzz-6-resize.png" style="max-width: 18%;" />
</div>
```
Key improvements and SEO considerations:

*   **Concise and Engaging Hook:** The first sentence now clearly states the core function and benefit.
*   **Targeted Keywords:** Included terms like "offline audio transcription," "audio translation," "OpenAI Whisper," and platform names (macOS, Windows, Linux) to improve searchability.
*   **Clear Headings and Structure:** Uses H2 headings for better organization and readability for both users and search engines.
*   **Bulleted Key Features:** Highlights the main selling points in an easily digestible format.
*   **Emphasis on Benefits:** The description focuses on what Buzz *does* for the user (transcribe and translate, offline, privacy, etc.)
*   **Installation Instructions:** Kept the instructions, making it easier for users to get started.
*   **Mac App Store Callout:** Highlighted the native Mac app version.
*   **Link to Original Repo:**  Added a clear link back to the GitHub repository.