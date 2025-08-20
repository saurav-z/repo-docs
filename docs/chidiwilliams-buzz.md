# Buzz: Offline Audio Transcription and Translation

**Effortlessly transcribe and translate audio on your computer with Buzz, powered by the powerful Whisper model.** 

[Get Buzz on the Mac App Store](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200) | [Documentation](https://chidiwilliams.github.io/buzz/) | [Original Repository](https://github.com/chidiwilliams/buzz)

Buzz leverages the cutting-edge capabilities of OpenAI's Whisper to provide accurate and efficient audio transcription and translation, all done locally on your device.

## Key Features

*   **Offline Processing:** Transcribe and translate audio without an internet connection, ensuring privacy and speed.
*   **Multi-Platform Support:** Available on macOS, Windows, and Linux.
*   **OpenAI Whisper Integration:**  Utilizes the advanced speech recognition and translation capabilities of OpenAI's Whisper.
*   **Mac App Store Version:** Experience Buzz with a streamlined, Mac-native interface, including features like audio playback, drag-and-drop import, transcript editing, and search.
*   **GPU Support (PyPI):**  Accelerate transcription on Windows with GPU support for NVIDIA GPUs via PyTorch.

## Installation

Choose your preferred method for installing Buzz:

### macOS

*   **Homebrew:**
    ```bash
    brew install --cask buzz
    ```
*   **Direct Download:** Download the `.dmg` file from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   **Executable Download:** Download and run the `.exe` file from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest). *Note: The app is not signed, you may need to select "More info" -> "Run anyway" during installation.*
*   **Winget:**
    ```bash
    winget install ChidiWilliams.Buzz
    ```
*   **GPU Support (PyPI):** Install the PyPI version for GPU acceleration:

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

Install dependencies and Buzz:

```bash
pip install buzz-captions
python -m buzz
```

**Important:** Ensure you have [ffmpeg](https://www.ffmpeg.org/download.html) installed.

### Latest Development Version

For access to the newest features and bug fixes, refer to the [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version).

## Screenshots

[Include Screenshots from original README, such as the import, main screen, preferences, etc.]