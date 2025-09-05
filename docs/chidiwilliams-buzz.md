# Buzz: Offline Audio Transcription and Translation

**Buzz is a powerful, offline audio transcription and translation tool, powered by OpenAI's Whisper, that puts you in control of your audio.** [Visit the original repository](https://github.com/chidiwilliams/buzz)

[![MIT License](https://img.shields.io/badge/license-MIT-green)](https://github.com/chidiwilliams/buzz)
[![CI](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml/badge.svg)](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/chidiwilliams/buzz/branch/main/graph/badge.svg?token=YJSB8S2VEP)](https://codecov.io/github/chidiwilliams/buzz)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/chidiwilliams/buzz)
[![Github all releases](https://img.shields.io/github/downloads/chidiwilliams/buzz/total.svg)](https://GitHub.com/chidiwilliams/buzz/releases/)

![Buzz](./buzz/assets/buzz-banner.jpg)

## Key Features

*   **Offline Transcription:** Transcribe audio files without an internet connection, ensuring privacy and speed.
*   **Translation Capabilities:** Translate your transcriptions into various languages.
*   **Powered by Whisper:** Utilizes the cutting-edge speech recognition technology of OpenAI's Whisper.
*   **Cross-Platform Compatibility:** Available for macOS, Windows, and Linux.
*   **Multiple Installation Options:** Install via PyPI, Brew, Winget, Flatpak, or Snap.
*   **Mac App Store Version:**  For an enhanced experience, explore the Mac-native version with additional features:
    *   Cleaner Interface
    *   Audio Playback
    *   Drag-and-Drop Import
    *   Transcript Editing
    *   Search Functionality

    [Download Buzz Captions on the Mac App Store](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200)

## Installation

### PyPI

1.  Install [ffmpeg](https://www.ffmpeg.org/download.html)
2.  Install Buzz: `pip install buzz-captions`
3.  Run Buzz: `python -m buzz`

### macOS

*   **Homebrew:** `brew install --cask buzz`
*   **Direct Download:** Download the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   **Direct Download:** Download and run the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest). *Note: The app is not signed; you may need to select "More info" -> "Run anyway" during installation.*
*   **Winget:** `winget install ChidiWilliams.Buzz`

**GPU Support (PyPI on Windows)**

For Nvidia GPU support:

1.  Install CUDA support for [torch](https://pytorch.org/get-started/locally/)
2.  Run the following `pip3 install` commands (see original README for specifics).

### Linux

*   **Flatpak:** `flatpak install flathub io.github.chidiwilliams.Buzz`
*   **Snap:** Follow the instructions in the original README.

### Latest Development Version

For the most recent features and bug fixes, refer to the [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version) for instructions on getting the latest development version.

### Screenshots

<div style="display: flex; flex-wrap: wrap;">
    <img alt="File import" src="share/screenshots/buzz-1-import.png" style="max-width: 18%; margin-right: 1%;" />
    <img alt="Main screen" src="share/screenshots/buzz-2-main_screen.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Preferences" src="share/screenshots/buzz-3-preferences.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Model preferences" src="share/screenshots/buzz-3.2-model-preferences.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Transcript" src="share/screenshots/buzz-4-transcript.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Live recording" src="share/screenshots/buzz-5-live_recording.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Resize" src="share/screenshots/buzz-6-resize.png" style="max-width: 18%;" />
</div>