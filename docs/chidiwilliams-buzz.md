# Buzz: Offline Audio Transcription & Translation Powered by Whisper

**Effortlessly transcribe and translate audio files on your computer with Buzz, leveraging the power of OpenAI's Whisper.** Explore the original repository on [GitHub](https://github.com/chidiwilliams/buzz).

## Key Features

*   **Offline Transcription:** Transcribe audio without an internet connection, ensuring privacy and speed.
*   **Multi-Platform Support:** Available for macOS, Windows, and Linux.
*   **Translation Capabilities:** Translate audio into multiple languages.
*   **Powered by Whisper:** Utilizes the robust speech recognition capabilities of OpenAI's Whisper.
*   **Easy Installation:** Simple setup via PyPI, brew, winget, Flatpak, and Snap.
*   **GPU Support:**  Utilize your NVIDIA GPU for faster processing (PyPI installation - Windows).

## Installation

Choose your operating system to install Buzz:

### macOS

*   **Brew:**
    ```shell
    brew install --cask buzz
    ```
*   **Direct Download:** Download the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   **Direct Download:** Download and run the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).  (Note: You may need to select `More info` -> `Run anyway` due to the app not being signed).
*   **Winget:**
    ```shell
    winget install ChidiWilliams.Buzz
    ```

**GPU Support for PyPI**

To enable GPU acceleration (Nvidia GPUs):

1.  Ensure CUDA support for [torch](https://pytorch.org/get-started/locally/) 
    ```
    pip3 install -U torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
    ```
2.  Install NVIDIA CUDA toolkit components:
    ```
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

## Additional Resources

*   [Documentation](https://chidiwilliams.github.io/buzz/)
*   [Buzz Captions on the Mac App Store](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200)
*   [FAQ (for development version)](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version)

## Screenshots

*(Screenshots would go here, as in the original README)*