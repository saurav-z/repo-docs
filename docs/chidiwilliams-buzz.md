# Buzz: Offline Audio Transcription and Translation

**Buzz is a powerful, offline audio transcription and translation tool, leveraging the accuracy of OpenAI's Whisper model.** [View the original repository](https://github.com/chidiwilliams/buzz)

**Key Features:**

*   **Offline Functionality:** Transcribe and translate audio directly on your computer, ensuring privacy and speed.
*   **OpenAI Whisper Powered:** Utilizes the state-of-the-art Whisper model for accurate and reliable transcriptions.
*   **Cross-Platform Support:** Available on macOS, Windows, and Linux, with installation options via PyPI, brew, winget, Flatpak, and Snap.
*   **Multiple Installation Methods:** Install Buzz using package managers (pip, brew, winget, Flatpak, Snap) or by downloading the executable.
*   **GPU Support:** Enhanced performance with GPU acceleration for NVIDIA GPUs (PyPI install).
*   **Mac App Store Version:** Get a Mac-native version with advanced features like audio playback, drag-and-drop import, and transcript editing.
*   **Ongoing Development:** Active development with a [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version) to keep you updated.

**Installation:**

*   **PyPI:**
    1.  Install [ffmpeg](https://www.ffmpeg.org/download.html).
    2.  `pip install buzz-captions`
    3.  `python -m buzz`
    4.  For GPU support on Windows, follow the instructions in the original README.

*   **macOS:**
    *   `brew install --cask buzz`
    *   Alternatively, download the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

*   **Windows:**
    *   Download and run the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).
    *   Or, install with `winget install ChidiWilliams.Buzz`

*   **Linux:**
    *   **Flatpak:** `flatpak install flathub io.github.chidiwilliams.Buzz`
    *   **Snap:**
        1.  `sudo apt-get install libportaudio2 libcanberra-gtk-module libcanberra-gtk3-module`
        2.  `sudo snap install buzz`
        3.  `sudo snap connect buzz:password-manager-service`

**Screenshots:**
*(Screenshots from the original README)*