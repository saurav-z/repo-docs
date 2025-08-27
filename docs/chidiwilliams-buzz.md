# Buzz: Offline Audio Transcription and Translation

**Effortlessly transcribe and translate audio on your computer with Buzz, powered by OpenAI's Whisper.**  Check out the [original repo](https://github.com/chidiwilliams/buzz) for more information.

## Key Features

*   **Offline Transcription:** Transcribe audio files without needing an internet connection, ensuring privacy and speed.
*   **Translation Capabilities:** Translate your transcriptions into multiple languages.
*   **Cross-Platform Support:**  Available for macOS, Windows, and Linux, offering flexibility for all users.
*   **GPU Acceleration:**  Leverage your NVIDIA GPU for faster processing (PyPI installation).
*   **Multiple Installation Methods:** Install via PyPI, brew, winget, Flatpak, or Snap, catering to different user preferences.
*   **Mac App Store Version:**  A feature-rich, native macOS version with enhanced capabilities is available.

## Installation

Choose your preferred method below:

### PyPI

1.  Install [ffmpeg](https://www.ffmpeg.org/download.html).
2.  Install Buzz:
    ```shell
    pip install buzz-captions
    python -m buzz
    ```

**Enable GPU Support for PyPI (Nvidia GPUs on Windows)**

Ensure CUDA support for [torch](https://pytorch.org/get-started/locally/)

```
pip3 install -U torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
pip3 install nvidia-cublas-cu12==12.8.3.14 nvidia-cuda-cupti-cu12==12.8.57 nvidia-cuda-nvrtc-cu12==12.8.61 nvidia-cuda-runtime-cu12==12.8.57 nvidia-cudnn-cu12==9.7.1.26 nvidia-cufft-cu12==11.3.3.41 nvidia-curand-cu12==10.3.9.55 nvidia-cusolver-cu12==11.7.2.55 nvidia-cusparse-cu12==12.5.4.2 nvidia-cusparselt-cu12==0.6.3 nvidia-nvjitlink-cu12==12.8.61 nvidia-nvtx-cu12==12.8.55 --extra-index-url https://pypi.ngc.nvidia.com
```

### macOS

*   **Homebrew:**
    ```shell
    brew install --cask buzz
    ```
*   **Download:** From the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   **Download:**  From the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).  *(Note:  You may receive a warning about the unsigned app. Select 'More info' -> 'Run anyway'.)*
*   **Winget:**
    ```shell
    winget install ChidiWilliams.Buzz
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

For the latest features and bug fixes, refer to the [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version).

## Additional Resources

*   [Documentation](https://chidiwilliams.github.io/buzz/)
*   [Buzz Captions on the Mac App Store](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200)

## Screenshots

<div style="display: flex; flex-wrap: wrap;">
    <img alt="File import" src="share/screenshots/buzz-1-import.png" style="max-width: 18%; margin-right: 1%;" />
    <img alt="Main screen" src="share/screenshots/buzz-2-main_screen.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Preferences" src="share/screenshots/buzz-3-preferences.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Model preferences" src="share/screenshots/buzz-3.2-model-preferences.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Transcript" src="share/screenshots/buzz-4-transcript.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Live recording" src="share/screenshots/buzz-5-live_recording.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Resize" src="share/screenshots/buzz-6-resize.png" style="max-width: 18%;" />
</div>