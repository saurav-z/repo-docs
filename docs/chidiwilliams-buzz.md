# Buzz: Offline Audio Transcription and Translation Powered by Whisper

**Buzz** empowers you to transcribe and translate audio effortlessly on your computer using the power of OpenAI's Whisper. ([See the original repo](https://github.com/chidiwilliams/buzz))

**Key Features:**

*   **Offline Functionality:** Transcribe and translate audio without an internet connection, ensuring privacy and efficiency.
*   **Whisper Integration:** Leverages the robust speech recognition capabilities of OpenAI's Whisper.
*   **Cross-Platform Support:** Available for macOS, Windows, and Linux.
*   **Multiple Installation Options:** Install via PyPI, Homebrew (macOS), winget (Windows), Flatpak (Linux), or Snap (Linux).
*   **GPU Acceleration:** Supports GPU acceleration for faster processing (requires CUDA setup for PyPI on Windows).
*   **User-Friendly Interface:** Clean and intuitive interface for seamless audio processing.
*   **[Mac App Store Version Available](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200)**: Get a Mac-native version with added features like audio playback, editing, and search.

## Installation

Choose your operating system for installation instructions:

### macOS

*   **Homebrew:**

    ```shell
    brew install --cask buzz
    ```
*   **Direct Download:** Download the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   **Direct Download:** Download and run the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest). (Note: App is not signed; you may need to select `More info` -> `Run anyway` during installation).
*   **Winget:**

    ```shell
    winget install ChidiWilliams.Buzz
    ```

*   **GPU Support (PyPI):** For GPU acceleration with an Nvidia GPU, install CUDA-compatible PyTorch and related packages:

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

### PyPI

*   Install [ffmpeg](https://www.ffmpeg.org/download.html)
*   Install Buzz:

    ```shell
    pip install buzz-captions
    python -m buzz
    ```

### Latest Development Version

For information on how to get the latest development version with the latest features and bug fixes, see the [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version).

## Screenshots

**(Screenshots of the app's interface are included here, as in the original README.)**