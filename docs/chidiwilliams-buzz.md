# Buzz: Offline Audio Transcription and Translation with Whisper

**Effortlessly transcribe and translate audio offline on your computer with Buzz, powered by OpenAI's Whisper!**

**(Check out the original repo: [https://github.com/chidiwilliams/buzz](https://github.com/chidiwilliams/buzz))**

[![MIT License](https://img.shields.io/badge/license-MIT-green)](https://github.com/chidiwilliams/buzz/blob/main/LICENSE)
[![CI](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml/badge.svg)](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/chidiwilliams/buzz/branch/main/graph/badge.svg?token=YJSB8S2VEP)](https://codecov.io/github/chidiwilliams/buzz)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/chidiwilliams/buzz)
[![Github all releases](https://img.shields.io/github/downloads/chidiwilliams/buzz/total.svg)](https://GitHub.com/chidiwilliams/buzz/releases/)

## Key Features

*   **Offline Transcription:** Transcribe audio files directly on your computer without needing an internet connection, ensuring privacy and speed.
*   **Offline Translation:** Translate transcriptions into multiple languages.
*   **Powered by OpenAI Whisper:** Leveraging the powerful and accurate speech recognition capabilities of OpenAI's Whisper model.
*   **Cross-Platform Compatibility:** Available for macOS, Windows, and Linux.

## Installation

### macOS

*   **Homebrew:** `brew install --cask buzz`
*   **Direct Download:** Download the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).
*   **Mac App Store:** Get a Mac-native version of Buzz with a cleaner look, audio playback, drag-and-drop import, transcript editing, search, and much more.

    [<img src="https://toolbox.marketingtools.apple.com/api/badges/download-on-the-mac-app-store/black/en-us?size=250x83&releaseDate=1679529600" alt="Download on the Mac App Store" />](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200)

### Windows

*   **Direct Download:** Download and run the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).
    *   *Note: The app is not signed; you may get a warning. Select "More info" -> "Run anyway."*
*   **Winget:** `winget install ChidiWilliams.Buzz`
*   **GPU Support for PyPI:**  To enable GPU support for Nvidia GPUs on Windows with a PyPI installation, follow the instructions in the original README to install the required PyTorch and CUDA libraries.

### Linux

*   **Flatpak:**  `flatpak install flathub io.github.chidiwilliams.Buzz`
*   **Snap:**  Follow the instructions in the original README to install the necessary dependencies and the Snap package.

### PyPI

1.  Install [ffmpeg](https://www.ffmpeg.org/download.html)
2.  Install Buzz: `pip install buzz-captions`
3.  Run Buzz: `python -m buzz`

### Latest Development Version

For the most up-to-date features and bug fixes, see the [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version).

## Documentation

Comprehensive documentation is available at: [https://chidiwilliams.github.io/buzz/](https://chidiwilliams.github.io/buzz/)

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