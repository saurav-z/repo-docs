# Buzz: Offline Audio Transcription and Translation

**Effortlessly transcribe and translate audio files on your computer with Buzz, powered by OpenAI's Whisper.**  ([See the original repository](https://github.com/chidiwilliams/buzz))

[![MIT License](https://img.shields.io/badge/license-MIT-green)](https://github.com/chidiwilliams/buzz)
[![CI](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml/badge.svg)](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/chidiwilliams/buzz/branch/main/graph/badge.svg?token=YJSB8S2VEP)](https://codecov.io/github/chidiwilliams/buzz)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/chidiwilliams/buzz)
[![Github all releases](https://img.shields.io/github/downloads/chidiwilliams/buzz/total.svg)](https://GitHub.com/chidiwilliams/buzz/releases/)

[Documentation](https://chidiwilliams.github.io/buzz/)

**Get the Mac-native version of Buzz with enhanced features on the [Mac App Store](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200).**

![Download on the Mac App Store](https://toolbox.marketingtools.apple.com/api/badges/download-on-the-mac-app-store/black/en-us?size=250x83&releaseDate=1679529600)

![Buzz](./buzz/assets/buzz-banner.jpg)

## Key Features

*   **Offline Transcription:** Transcribe audio files directly on your computer, ensuring privacy and speed.
*   **Multi-Language Support:**  Translate audio into different languages.
*   **Powered by Whisper:** Utilizes the powerful OpenAI Whisper model for accurate transcription.
*   **Cross-Platform:** Available on macOS, Windows, and Linux.
*   **Easy Installation:**  Install through PyPI, brew, winget, Flatpak, or Snap.
*   **Mac App Store Version:** Enjoy a Mac-native app with advanced features.

## Installation

Choose your operating system:

### PyPI

1.  Install [ffmpeg](https://www.ffmpeg.org/download.html).
2.  Install Buzz:

```shell
pip install buzz-captions
python -m buzz
```

### macOS

Install using [brew](https://brew.sh/):

```shell
brew install --cask buzz
```

Or download the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

Download and run the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest). *Note:  You may need to select `More info` -> `Run anyway` during installation due to the app not being signed.*

**Alternative - Install with [winget](https://learn.microsoft.com/en-us/windows/package-manager/winget/)**

```shell
winget install ChidiWilliams.Buzz
```

**GPU support for PyPI**

To enable GPU support for Nvidia GPUs on Windows, ensure CUDA support for [torch](https://pytorch.org/get-started/locally/) is installed:

```
pip3 install -U torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
pip3 install nvidia-cublas-cu12==12.8.3.14 nvidia-cuda-cupti-cu12==12.8.57 nvidia-cuda-nvrtc-cu12==12.8.61 nvidia-cuda-runtime-cu12==12.8.57 nvidia-cudnn-cu12==9.7.1.26 nvidia-cufft-cu12==11.3.3.41 nvidia-curand-cu12==10.3.9.55 nvidia-cusolver-cu12==11.7.2.55 nvidia-cusparse-cu12==12.5.4.2 nvidia-cusparselt-cu12==0.6.3 nvidia-nvjitlink-cu12==12.8.61 nvidia-nvtx-cu12==12.8.55 --extra-index-url https://pypi.ngc.nvidia.com
```

### Linux

Buzz is available as a [Flatpak](https://flathub.org/apps/io.github.chidiwilliams.Buzz) or a [Snap](https://snapcraft.io/buzz).

**Flatpak Installation:**

```shell
flatpak install flathub io.github.chidiwilliams.Buzz
```

**Snap Installation:**

```shell
sudo apt-get install libportaudio2 libcanberra-gtk-module libcanberra-gtk3-module
sudo snap install buzz
sudo snap connect buzz:password-manager-service
```

## Latest Development Version

For access to the latest features and bug fixes, see the [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version).

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
```

Key improvements and explanations:

*   **SEO-Friendly Title:**  Uses "Offline Audio Transcription and Translation" to target relevant search terms.
*   **One-Sentence Hook:**  A clear, concise sentence that immediately describes the core functionality.
*   **Bulleted Key Features:**  Highlights the main benefits in an easily scannable format.
*   **Clear Headings:**  Organizes the information logically (Installation, Key Features, etc.).
*   **Mac App Store Mention:**  Prominently features the Mac App Store version with a direct link.
*   **Installation Instructions:** Improved readability.
*   **Links Back to Repo:** Includes a prominent link back to the original GitHub repository.
*   **Markdown formatting** makes it visually appealing and easier to read.
*   **Alt text** added for images for accessibility.
*   **Removed the language link** since it is not necessary for the primary README.