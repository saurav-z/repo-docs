# Buzz: Offline Audio Transcription & Translation 

**Effortlessly transcribe and translate audio on your computer with Buzz, powered by OpenAI's Whisper.**

[View the original repository on GitHub](https://github.com/chidiwilliams/buzz)

Buzz is a powerful, cross-platform application that allows you to transcribe and translate audio files offline, using the robust capabilities of OpenAI's Whisper. Available on macOS, Windows, and Linux.

**Key Features:**

*   **Offline Transcription:** Process audio files directly on your computer, ensuring privacy and speed.
*   **Multi-Platform Support:** Available for macOS, Windows, and Linux, providing flexibility for all users.
*   **Translation Capabilities:** Translate audio into multiple languages.
*   **Powered by OpenAI Whisper:** Leverages the state-of-the-art speech recognition technology.
*   **Multiple Installation Options:** Install via PyPI, macOS (Homebrew & DMG), Windows (exe & winget), and Linux (Flatpak & Snap).
*   **GPU Support (PyPI):** Leverage the power of your Nvidia GPU for faster transcription on Windows.

**Installation:**

Choose your operating system for installation instructions:

*   **macOS:**
    *   **Homebrew:** `brew install --cask buzz`
    *   **DMG:** Download the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

*   **Windows:**
    *   **Executable:** Download the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest). You may need to select `More info` -> `Run anyway` during installation due to the app not being signed.
    *   **Winget:** `winget install ChidiWilliams.Buzz`
    *   **GPU Support (PyPI):** See the original README for detailed GPU support instructions for PyPI installations.

*   **Linux:**
    *   **Flatpak:** `flatpak install flathub io.github.chidiwilliams.Buzz`
    *   **Snap:** `sudo snap install buzz` (requires additional setup - see the original README)

*   **PyPI:**
    1.  Install [ffmpeg](https://www.ffmpeg.org/download.html)
    2.  `pip install buzz-captions`
    3.  `python -m buzz`

**Screenshots:** (Screenshots from original README)

<div style="display: flex; flex-wrap: wrap;">
    <img alt="File import" src="share/screenshots/buzz-1-import.png" style="max-width: 18%; margin-right: 1%;" />
    <img alt="Main screen" src="share/screenshots/buzz-2-main_screen.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Preferences" src="share/screenshots/buzz-3-preferences.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Model preferences" src="share/screenshots/buzz-3.2-model-preferences.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Transcript" src="share/screenshots/buzz-4-transcript.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Live recording" src="share/screenshots/buzz-5-live_recording.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Resize" src="share/screenshots/buzz-6-resize.png" style="max-width: 18%;" />
</div>

**For the latest development version and more information, see the [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version).**

**Additional Resources:**

*   [Documentation](https://chidiwilliams.github.io/buzz/)
*   [Buzz Captions on the App Store](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200) (Mac-native version with additional features)