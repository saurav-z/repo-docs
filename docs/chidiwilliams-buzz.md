# Buzz: Offline Audio Transcription and Translation

**Buzz empowers you to transcribe and translate audio files directly on your computer, harnessing the power of OpenAI's Whisper.** ([Original Repository](https://github.com/chidiwilliams/buzz))

**Key Features:**

*   **Offline Transcription & Translation:** Process audio without an internet connection, ensuring privacy and speed.
*   **Powered by Whisper:** Utilizing the cutting-edge open-source speech recognition model from OpenAI.
*   **Cross-Platform Compatibility:** Available for macOS, Windows, and Linux.
*   **Multiple Installation Options:** Easily install via PyPI, brew, winget, Flatpak, or Snap.
*   **GPU Acceleration (PyPI):** Take advantage of your NVIDIA GPU for faster processing (Windows).
*   **Mac App Store Version:** For a superior experience, explore the Mac-native version with enhanced features like audio playback, editing, and search.
*   **Regular Updates & Active Development:** Benefit from bug fixes and the latest features by accessing development versions.

**Installation:**

Choose your operating system below:

*   **macOS:**
    *   **Homebrew:** `brew install --cask buzz`
    *   **Releases:** Download the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).
*   **Windows:**
    *   **Winget:** `winget install ChidiWilliams.Buzz`
    *   **Releases:** Download the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest). (Note: App is not signed and may require you to bypass a warning.)
    *   **GPU Support (PyPI):** For GPU support on Windows, ensure CUDA support for [torch](https://pytorch.org/get-started/locally/) and install the necessary packages.
*   **Linux:**
    *   **Flatpak:** `flatpak install flathub io.github.chidiwilliams.Buzz`
    *   **Snap:** `sudo apt-get install libportaudio2 libcanberra-gtk-module libcanberra-gtk3-module; sudo snap install buzz; sudo snap connect buzz:password-manager-service`
*   **PyPI:**
    1.  Install [ffmpeg](https://www.ffmpeg.org/download.html).
    2.  `pip install buzz-captions`
    3.  `python -m buzz`

**Screenshots:**

[Include screenshot images here]

**Additional Resources:**

*   [Documentation](https://chidiwilliams.github.io/buzz/)
*   [Buzz Captions on the App Store](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200)
*   [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version)
*   [Original Repository](https://github.com/chidiwilliams/buzz)

**Note:** This README offers a concise overview.  For detailed information, consult the documentation.