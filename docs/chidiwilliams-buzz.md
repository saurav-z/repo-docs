# Buzz: Offline Audio Transcription and Translation Powered by Whisper

**Effortlessly transcribe and translate audio on your computer with Buzz, a powerful tool leveraging OpenAI's Whisper technology.**  [Check out the original repository](https://github.com/chidiwilliams/buzz)

*   **Offline Functionality:** Transcribe and translate audio without an internet connection.
*   **Powered by Whisper:** Utilizes the robust and accurate OpenAI Whisper model.
*   **Cross-Platform:** Available for macOS, Windows, and Linux.
*   **Multiple Installation Options:** Install via PyPI, Homebrew, Winget, Flatpak, or Snap.
*   **GPU Support:**  Leverage GPU acceleration for faster processing (with appropriate setup).
*   **Mac App Store Version:** For an enhanced experience with audio playback, editing, and more, explore the dedicated Mac App Store version: [![Download on the Mac App Store](https://toolbox.marketingtools.apple.com/api/badges/download-on-the-mac-app-store/black/en-us?size=250x83&releaseDate=1679529600)](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200)
*   **Regular Updates:** Get the latest features and bug fixes with the development version, available via the [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version).
*   **Easy to Use:** Simple installation and intuitive interface for quick transcription.

## Installation

Choose your preferred method:

### PyPI

1.  Install [ffmpeg](https://www.ffmpeg.org/download.html).
2.  Install Buzz:

    ```bash
    pip install buzz-captions
    python -m buzz
    ```
    **GPU Support (Nvidia):** For optimal performance on Nvidia GPUs, install CUDA support for torch, as explained in the original README.

### macOS

*   **Homebrew:**

    ```bash
    brew install --cask buzz
    ```
*   **Releases:** Download the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   **Releases:** Download and run the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest). You may need to select "More info" -> "Run anyway" if you encounter a security warning.
*   **Winget:**

    ```bash
    winget install ChidiWilliams.Buzz
    ```
    **GPU Support (Nvidia):** Install CUDA support for torch, as explained in the original README.

### Linux

*   **Flatpak:**

    ```bash
    flatpak install flathub io.github.chidiwilliams.Buzz
    ```
*   **Snap:**

    ```bash
    sudo apt-get install libportaudio2 libcanberra-gtk-module libcanberra-gtk3-module
    sudo snap install buzz
    sudo snap connect buzz:password-manager-service
    ```

## Screenshots

\[Screenshots from original README]