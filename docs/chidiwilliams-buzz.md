# Buzz: Offline Audio Transcription and Translation

**Buzz empowers you to transcribe and translate audio directly on your computer, leveraging the power of OpenAI's Whisper.** ([Original Repository](https://github.com/chidiwilliams/buzz))

[![MIT License](https://img.shields.io/badge/license-MIT-green)](https://github.com/chidiwilliams/buzz)
[![CI](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml/badge.svg)](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/chidiwilliams/buzz/branch/main/graph/badge.svg?token=YJSB8S2VEP)](https://codecov.io/github/chidiwilliams/buzz)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/chidiwilliams/buzz)
[![Github all releases](https://img.shields.io/github/downloads/chidiwilliams/buzz/total.svg)](https://GitHub.com/chidiwilliams/buzz/releases/)

## Key Features

*   **Offline Transcription & Translation:** Process audio files locally, ensuring privacy and speed.
*   **Powered by Whisper:** Utilizes the robust speech-to-text capabilities of OpenAI's Whisper model.
*   **Multi-Platform Support:** Available for macOS, Windows, and Linux.
*   **Multiple Installation Options:** Install via PyPI, Homebrew, Winget, Flatpak, or Snap.
*   **GPU Acceleration:**  Take advantage of GPU support for faster processing on compatible hardware.
*   **[Mac App Store Version](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200):** Get a Mac-native version of Buzz with additional features such as audio playback, transcript editing, search, drag-and-drop import, and more.

## Installation

### macOS

*   **Homebrew:** `brew install --cask buzz`
*   **Direct Download:**  Download the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   **Direct Download:** Download and run the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).  You may need to select `More info` -> `Run anyway` if you get a warning.
*   **Winget:** `winget install ChidiWilliams.Buzz`
*   **GPU Support:**  See instructions in the original README for CUDA setup.

### Linux

*   **Flatpak:** `flatpak install flathub io.github.chidiwilliams.Buzz`
*   **Snap:** See the original README for installation instructions.

### PyPI

*   Install ffmpeg:  Follow instructions on [ffmpeg website](https://www.ffmpeg.org/download.html)
*   Install Buzz: `pip install buzz-captions`
*   Run Buzz: `python -m buzz`
*   **GPU Support:** See instructions in the original README for CUDA setup.

### Latest Development Version

For information on how to get the latest development version with new features and bug fixes, see the [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version).

## Documentation

Detailed documentation can be found [here](https://chidiwilliams.github.io/buzz/).

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