# Buzz: Offline Audio Transcription & Translation Powered by Whisper

**Effortlessly transcribe and translate audio on your computer with Buzz, a powerful tool leveraging the capabilities of OpenAI's Whisper model.** ([Original Repository](https://github.com/chidiwilliams/buzz))

## Key Features:

*   **Offline Transcription:** Transcribe audio files directly on your device, ensuring privacy and speed.
*   **Multi-Platform Support:** Available for macOS, Windows, and Linux, providing flexibility for all users.
*   **Translation Capabilities:** Translate your audio into multiple languages.
*   **Powered by Whisper:** Utilizes the cutting-edge OpenAI Whisper model for accurate results.
*   **Multiple Installation Options:** Install via PyPI, Brew, Winget, Flatpak, or Snap for convenient access.
*   **GPU Support:** Accelerate transcription speeds on Windows with Nvidia GPUs (requires specific CUDA setup).

## Installation

Choose your preferred installation method below:

### PyPI

1.  Install [ffmpeg](https://www.ffmpeg.org/download.html)
2.  Install Buzz:
    ```shell
    pip install buzz-captions
    python -m buzz
    ```

### macOS

*   Install via [Homebrew](https://brew.sh/):
    ```shell
    brew install --cask buzz
    ```

*   Or download the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   Download and run the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).
    *   *Note:* App is not signed, you will get a warning when you install it. Select `More info` -> `Run anyway`.*

*   **Alternatively, install with [winget](https://learn.microsoft.com/en-us/windows/package-manager/winget/)**:
    ```shell
    winget install ChidiWilliams.Buzz
    ```

*   **GPU Support for PyPI (Nvidia):**
    For GPU acceleration, ensure CUDA support for [PyTorch](https://pytorch.org/get-started/locally/) and install the necessary packages:
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

### Latest Development Version

For the latest features and bug fixes, see the [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version).

### Screenshots

**(Image gallery - refer to original README for images.)**

---