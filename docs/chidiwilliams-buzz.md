# Buzz: Offline Audio Transcription and Translation

**Effortlessly transcribe and translate audio files on your computer with Buzz, powered by OpenAI's Whisper.**  ([Original Repository](https://github.com/chidiwilliams/buzz))

Buzz utilizes the power of OpenAI's Whisper to provide accurate and efficient transcription and translation directly on your machine. This means you can transcribe audio without an internet connection, protecting your privacy and saving time.

## Key Features

*   **Offline Transcription & Translation:** Transcribe and translate audio files locally, ensuring privacy and speed.
*   **Cross-Platform Compatibility:** Available for macOS, Windows, and Linux (Flatpak & Snap).
*   **Powered by OpenAI Whisper:** Utilizing the cutting-edge speech recognition technology from OpenAI.
*   **Multiple Installation Options:** Install via PyPI, macOS (Brew, DMG), Windows (EXE, Winget), and Linux (Flatpak, Snap).
*   **GPU Acceleration:** Supports GPU acceleration for faster processing on compatible systems (PyPI).
*   **Mac App Store Version:** Explore a Mac-native version on the [Mac App Store](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200) with additional features like transcript editing, search, and audio playback.

## Installation

Choose your preferred installation method:

### PyPI

1.  Install [ffmpeg](https://www.ffmpeg.org/download.html)
2.  Install Buzz:

    ```bash
    pip install buzz-captions
    python -m buzz
    ```
3.  **GPU Support (Nvidia):** To enable GPU support on Windows for the PyPI version, ensure CUDA is installed, then run:

    ```bash
    pip3 install -U torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
    pip3 install nvidia-cublas-cu12==12.8.3.14 nvidia-cuda-cupti-cu12==12.8.57 nvidia-cuda-nvrtc-cu12==12.8.61 nvidia-cuda-runtime-cu12==12.8.57 nvidia-cudnn-cu12==9.7.1.26 nvidia-cufft-cu12==11.3.3.41 nvidia-curand-cu12==10.3.9.55 nvidia-cusolver-cu12==11.7.2.55 nvidia-cusparse-cu12==12.5.4.2 nvidia-cusparselt-cu12==0.6.3 nvidia-nvjitlink-cu12==12.8.61 nvidia-nvtx-cu12==12.8.55 --extra-index-url https://pypi.ngc.nvidia.com
    ```

### macOS

*   **Homebrew:**

    ```bash
    brew install --cask buzz
    ```
*   **Download DMG:** Download the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   **Download EXE:** Download and run the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).  You may need to select `More info` -> `Run anyway` due to the app not being signed.
*   **Winget:**

    ```bash
    winget install ChidiWilliams.Buzz
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

## Latest Development Version

For the latest features and bug fixes, see the [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version) for info on how to get it.

## Screenshots

*(Screenshots from original README remain)*