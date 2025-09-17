# Buzz: Offline Audio Transcription & Translation

**Effortlessly transcribe and translate audio on your computer with Buzz, powered by OpenAI's Whisper.**  [Learn more](https://github.com/chidiwilliams/buzz)

Buzz utilizes the power of OpenAI's Whisper to provide fast and accurate audio transcription and translation directly on your computer, eliminating the need for internet connectivity and ensuring your audio data's privacy.

## Key Features:

*   **Offline Transcription:** Transcribe audio files without an internet connection, ensuring privacy and accessibility.
*   **Multilingual Support:** Supports transcription and translation in numerous languages.
*   **Cross-Platform Compatibility:** Available for macOS, Windows, and Linux.
*   **Multiple Installation Options:** Install via PyPI, brew (macOS), winget (Windows), Flatpak (Linux), or Snap (Linux).
*   **GPU Acceleration:**  Leverages GPU for faster processing (requires setup, see installation instructions).

## Installation

### PyPI

1.  Install [ffmpeg](https://www.ffmpeg.org/download.html)
2.  Install Buzz:

    ```shell
    pip install buzz-captions
    python -m buzz
    ```

### macOS

*   **Homebrew:**

    ```shell
    brew install --cask buzz
    ```

*   **Download:** Download the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   **Download:** Download and run the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).  You may need to bypass a security warning.
*   **winget:**

    ```shell
    winget install ChidiWilliams.Buzz
    ```

*   **GPU Support (PyPI):** For Nvidia GPUs, install CUDA support for [torch](https://pytorch.org/get-started/locally/).

    ```
    pip3 install -U torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
    pip3 install nvidia-cublas-cu12==12.8.3.14 nvidia-cuda-cupti-cu12==12.8.57 nvidia-cuda-nvrtc-cu12==12.8.61 nvidia-cuda-runtime-cu12==12.8.57 nvidia-cudnn-cu12==9.7.1.26 nvidia-cufft-cu12==11.3.3.41 nvidia-curand-cu12==10.3.9.55 nvidia-cusolver-cu12==11.7.2.55 nvidia-cusparse-cu12==12.5.4.2 nvidia-cusparselt-cu12==0.6.3 nvidia-nvjitlink-cu12==12.8.61 nvidia-nvtx-cu12==12.8.55 --extra-index-url https://pypi.ngc.nvidia.com
    ```

### Linux

*   **Flatpak:**

    ```shell
    flatpak install flathub io.github.chidiwilliams.Buzz
    ```

*   **Snap:**

    ```shell
    sudo apt-get install libportaudio2 libcanberra-gtk-module libcanberra-gtk3-module
    sudo snap install buzz
    sudo snap connect buzz:password-manager-service
    ```

## App Store Version

Buzz is also available as a native macOS app with additional features, including a cleaner look, audio playback, drag-and-drop import, transcript editing, and search.

<a href="https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&amp;itsct=apps_box_badge&amp;itscg=30200"><img src="https://toolbox.marketingtools.apple.com/api/badges/download-on-the-mac-app-store/black/en-us?size=250x83&amp;releaseDate=1679529600" alt="Download on the Mac App Store" /></a>

## Screenshots

(Screenshots of the application are included here)

## Development Version

For the latest features and bug fixes, consult the [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version).

## Contributing

Contributions are welcome! Please see the [contributing guidelines](CONTRIBUTING.md).

## License

Buzz is licensed under the MIT License.

---

**[Back to the project on GitHub](https://github.com/chidiwilliams/buzz)**