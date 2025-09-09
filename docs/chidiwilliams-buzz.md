# Buzz: Offline Audio Transcription and Translation Powered by Whisper

**Buzz** is a powerful, offline audio transcription and translation tool that utilizes the cutting-edge OpenAI Whisper model, giving you accurate results without relying on an internet connection. ([View the original repository](https://github.com/chidiwilliams/buzz))

[![MIT License](https://img.shields.io/badge/license-MIT-green)](https://github.com/chidiwilliams/buzz)
[![CI](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml/badge.svg)](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/chidiwilliams/buzz/branch/main/graph/badge.svg?token=YJSB8S2VEP)](https://codecov.io/github/chidiwilliams/buzz)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/chidiwilliams/buzz)
[![Github all releases](https://img.shields.io/github/downloads/chidiwilliams/buzz/total.svg)](https://GitHub.com/chidiwilliams/buzz/releases/)

[Documentation](https://chidiwilliams.github.io/buzz/) | [Buzz Captions on the App Store](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200)

## Key Features

*   **Offline Transcription:** Transcribe audio files without an internet connection using Whisper.
*   **Offline Translation:** Translate your transcriptions into various languages, all offline.
*   **Cross-Platform:** Available on macOS, Windows, and Linux.
*   **Multiple Installation Options:** Install via PyPI, brew, winget, Flatpak, or Snap.
*   **GPU Acceleration (Optional):** Leverage your GPU for faster processing (specific instructions for PyPI on Windows).

## Installation

Choose your operating system and preferred method:

### macOS

*   **Homebrew:** `brew install --cask buzz`
*   **Direct Download:** Get the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   **Direct Download:** Get the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).  Note: You may need to select "More info" -> "Run anyway" due to the app not being signed.
*   **Winget:** `winget install ChidiWilliams.Buzz`
*   **PyPI (with GPU support):**  Install with the commands listed in the original README.

### Linux

*   **Flatpak:** `flatpak install flathub io.github.chidiwilliams.Buzz`
*   **Snap:** Follow the instructions from the original README.

### PyPI

Install [ffmpeg](https://www.ffmpeg.org/download.html)

Install Buzz

```shell
pip install buzz-captions
python -m buzz
```

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