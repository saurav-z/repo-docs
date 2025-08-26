# Buzz: Offline Audio Transcription and Translation Powered by Whisper

**Buzz** is a powerful, open-source application that lets you transcribe and translate audio files offline on your personal computer, leveraging the cutting-edge capabilities of OpenAI's Whisper. Explore the original repository at [https://github.com/chidiwilliams/buzz](https://github.com/chidiwilliams/buzz).

## Key Features:

*   **Offline Transcription:** Transcribe audio files directly on your computer without an internet connection.
*   **Multi-Language Support:**  Translate audio into various languages.
*   **Cross-Platform:** Available on macOS, Windows, and Linux.
*   **Open Source:** Built upon the powerful OpenAI Whisper model.
*   **Multiple Installation Options:** Install via PyPI, Homebrew (macOS), winget (Windows), Flatpak (Linux), or Snap (Linux).
*   **GPU Acceleration:**  Utilizes GPU (Nvidia) for faster transcription on PyPI installed version.

## Installation

### PyPI

1.  Install [ffmpeg](https://www.ffmpeg.org/download.html)
2.  Install Buzz:

    ```shell
    pip install buzz-captions
    python -m buzz
    ```
### macOS

Install with [brew utility](https://brew.sh/)

```shell
brew install --cask buzz
```
Or download the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

Download and run the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

App is not signed, you will get a warning when you install it. Select `More info` -> `Run anyway`.

**Alternatively, install with [winget](https://learn.microsoft.com/en-us/windows/package-manager/winget/)**

```shell
winget install ChidiWilliams.Buzz
```

**GPU support for PyPI**

To have GPU support for Nvidia GPUS on Windows, for PyPI installed version ensure, CUDA support for [torch](https://pytorch.org/get-started/locally/) 

```
pip3 install -U torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
pip3 install nvidia-cublas-cu12==12.8.3.14 nvidia-cuda-cupti-cu12==12.8.57 nvidia-cuda-nvrtc-cu12==12.8.61 nvidia-cuda-runtime-cu12==12.8.57 nvidia-cudnn-cu12==9.7.1.26 nvidia-cufft-cu12==11.3.3.41 nvidia-curand-cu12==10.3.9.55 nvidia-cusolver-cu12==11.7.2.55 nvidia-cusparse-cu12==12.5.4.2 nvidia-cusparselt-cu12==0.6.3 nvidia-nvjitlink-cu12==12.8.61 nvidia-nvtx-cu12==12.8.55 --extra-index-url https://pypi.ngc.nvidia.com
```

### Linux

Buzz is available as a [Flatpak](https://flathub.org/apps/io.github.chidiwilliams.Buzz) or a [Snap](https://snapcraft.io/buzz). 

To install flatpak, run:
```shell
flatpak install flathub io.github.chidiwilliams.Buzz
```

To install snap, run:
```shell
sudo apt-get install libportaudio2 libcanberra-gtk-module libcanberra-gtk3-module
sudo snap install buzz
sudo snap connect buzz:password-manager-service
```

### Latest development version

For info on how to get latest development version with latest features and bug fixes see [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version).

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