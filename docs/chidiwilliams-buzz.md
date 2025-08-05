# Buzz: Offline Audio Transcription and Translation Powered by Whisper

Buzz is an open-source application that lets you transcribe and translate audio offline, directly on your computer.  [Learn more on the original repository](https://github.com/chidiwilliams/buzz).

[Documentation](https://chidiwilliams.github.io/buzz/) | [Buzz Captions on the App Store](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200)

**Key Features:**

*   **Offline Transcription & Translation:** Processes audio locally using OpenAI's Whisper, ensuring privacy and speed.
*   **Cross-Platform Support:** Available for macOS, Windows, and Linux.
*   **Multiple Installation Options:** Install via PyPI, macOS (Homebrew, DMG), Windows (exe, winget), Linux (Flatpak, Snap).
*   **GPU Support:**  Utilizes your GPU (Nvidia) for faster transcription (PyPI version).

**Get a Mac-native version of Buzz with additional features:**
[Download on the Mac App Store](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200)

![Buzz](./buzz/assets/buzz-banner.jpg)

## Installation

Choose your operating system for installation instructions:

### PyPI

1.  Install [ffmpeg](https://www.ffmpeg.org/download.html)
2.  Install Buzz

```shell
pip install buzz-captions
python -m buzz
```

### macOS

*   **Homebrew:**

    ```shell
    brew install --cask buzz
    ```

*   **DMG:** Download the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   **Executable:** Download and run the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).  *Note: You may receive a security warning; select "More info" -> "Run anyway".*

*   **Winget:**

    ```shell
    winget install ChidiWilliams.Buzz
    ```

**GPU Support for PyPI (Nvidia)**

Ensure CUDA support for [torch](https://pytorch.org/get-started/locally/) is enabled, then run:

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

<div style="display: flex; flex-wrap: wrap;">
    <img alt="File import" src="share/screenshots/buzz-1-import.png" style="max-width: 18%; margin-right: 1%;" />
    <img alt="Main screen" src="share/screenshots/buzz-2-main_screen.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Preferences" src="share/screenshots/buzz-3-preferences.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Transcript" src="share/screenshots/buzz-4-transcript.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Live recording" src="share/screenshots/buzz-5-live_recording.png" style="max-width: 18%;" />
</div>