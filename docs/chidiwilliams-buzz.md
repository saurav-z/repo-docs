# Buzz: Offline Audio Transcription and Translation

**Effortlessly transcribe and translate audio on your computer with Buzz, powered by OpenAI's Whisper.**

[View the original repository on GitHub](https://github.com/chidiwilliams/buzz)

Buzz allows you to transcribe audio files, translate them into different languages, and is available on multiple platforms.

## Key Features

*   **Offline Processing:** Transcribe and translate your audio files without an internet connection, ensuring privacy and speed.
*   **Powered by Whisper:** Utilizes the powerful and accurate OpenAI Whisper model for high-quality transcription and translation.
*   **Cross-Platform Compatibility:** Available for macOS, Windows, and Linux, offering flexibility for all users.
*   **Multiple Installation Options:** Install via PyPI, brew, winget, Flatpak, Snap, or download executable files for ease of use.
*   **GPU Acceleration:** Utilize your NVIDIA GPU for faster processing (PyPI installation).
*   **Mac App Store Version:** A native Mac version of Buzz is available on the App Store with additional features such as editing, search and audio playback.

## Installation

Choose your operating system and preferred installation method:

### macOS

*   **Homebrew:** `brew install --cask buzz`
*   **Download:** Download the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).
*   **Mac App Store:** Find the native Mac app [here](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200)

### Windows

*   **Winget:** `winget install ChidiWilliams.Buzz`
*   **Download:** Download the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).
*   **GPU Support (PyPI):**  For GPU support, see instructions within the original README.

### Linux

*   **Flatpak:** `flatpak install flathub io.github.chidiwilliams.Buzz`
*   **Snap:**  `sudo snap install buzz` (Additional setup steps are required; see original README).

### PyPI

1.  Install [ffmpeg](https://www.ffmpeg.org/download.html)
2.  Install Buzz:  `pip install buzz-captions`
3.  Run Buzz: `python -m buzz`

## Screenshots

[Include the screenshots from the original README here for a visual demonstration of the app.]

## Documentation

*   [Documentation](https://chidiwilliams.github.io/buzz/)
*   [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version)