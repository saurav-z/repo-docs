# Buzz: Offline Audio Transcription and Translation

**Effortlessly transcribe and translate audio files on your computer with Buzz, powered by OpenAI's Whisper.**  For a more polished experience with added features, check out the [Mac App Store version](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200).

[View the original repository on GitHub](https://github.com/chidiwilliams/buzz)

## Key Features

*   **Offline Transcription:** Transcribe audio files directly on your computer, ensuring privacy and speed.
*   **Offline Translation:** Translate audio into multiple languages without relying on an internet connection.
*   **Powered by Whisper:** Leverages the powerful speech-to-text capabilities of OpenAI's Whisper model.
*   **Cross-Platform Compatibility:** Available for macOS, Windows, and Linux.
*   **Multiple Installation Options:** Install via PyPI, brew, winget, Flatpak, or Snap.
*   **GPU Support:** Utilize your NVIDIA GPU for faster transcription on Windows (PyPI install).
*   **Regular Updates:** Benefit from ongoing improvements and bug fixes.

## Installation

Choose your operating system for detailed installation instructions:

### macOS

*   **Homebrew:**
    ```bash
    brew install --cask buzz
    ```
*   **Direct Download:** Download the `.dmg` file from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   **Winget:**
    ```bash
    winget install ChidiWilliams.Buzz
    ```
*   **Direct Download:** Download the `.exe` file from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).
    *   *Note:* Due to the app not being signed, you may see a warning during installation. Select "More info" -> "Run anyway."
*   **GPU Support (PyPI Install):** For NVIDIA GPU acceleration, ensure CUDA support for PyTorch:
    ```bash
    pip3 install -U torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
    pip3 install nvidia-cublas-cu12==12.8.3.14 nvidia-cuda-cupti-cu12==12.8.57 nvidia-cuda-nvrtc-cu12==12.8.61 nvidia-cuda-runtime-cu12==12.8.57 nvidia-cudnn-cu12==9.7.1.26 nvidia-cufft-cu12==11.3.3.41 nvidia-curand-cu12==10.3.9.55 nvidia-cusolver-cu12==11.7.2.55 nvidia-cusparse-cu12==12.5.4.2 nvidia-cusparselt-cu12==0.6.3 nvidia-nvjitlink-cu12==12.8.61 nvidia-nvtx-cu12==12.8.55 --extra-index-url https://pypi.ngc.nvidia.com
    ```

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

### PyPI

*   **Requirements:** Install [ffmpeg](https://www.ffmpeg.org/download.html).
*   **Installation:**
    ```shell
    pip install buzz-captions
    python -m buzz
    ```

## Latest Development Version

For the latest features and bug fixes, see the [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version) for information on accessing the development version.

## Screenshots

*(Include screenshots of the app's interface here)*