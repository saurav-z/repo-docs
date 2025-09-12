# Buzz: Offline Audio Transcription & Translation with OpenAI Whisper

**Effortlessly transcribe and translate audio files on your computer with Buzz, powered by the powerful OpenAI Whisper technology.**  [Explore the original repository](https://github.com/chidiwilliams/buzz).

## Key Features

*   **Offline Transcription:** Transcribe audio files directly on your computer, ensuring privacy and speed.
*   **Multi-Language Support:** Translate audio into numerous languages.
*   **Powered by OpenAI Whisper:** Leverage the cutting-edge audio processing capabilities of OpenAI's Whisper model.
*   **Cross-Platform:** Available for macOS, Windows, and Linux.
*   **Multiple Installation Options:** Install via PyPI, Homebrew, winget, Flatpak, or Snap.
*   **Mac App Store Version:**  Get a Mac-native version of Buzz with a cleaner look, audio playback, drag-and-drop import, transcript editing, search, and much more. [Download on the Mac App Store](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200)
*   **GPU Acceleration (for PyPI):** Enhance performance on Windows with GPU support for NVIDIA GPUs (requires CUDA setup).

## Installation

Choose your preferred method:

### PyPI

1.  Install [ffmpeg](https://www.ffmpeg.org/download.html)
2.  Install Buzz:
    ```shell
    pip install buzz-captions
    python -m buzz
    ```
3.  **Optional: GPU Support for PyPI (Windows with NVIDIA)**

    *   Ensure CUDA support for [torch](https://pytorch.org/get-started/locally/)
    ```
    pip3 install -U torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
    pip3 install nvidia-cublas-cu12==12.8.3.14 nvidia-cuda-cupti-cu12==12.8.57 nvidia-cuda-nvrtc-cu12==12.8.61 nvidia-cuda-runtime-cu12==12.8.57 nvidia-cudnn-cu12==9.7.1.26 nvidia-cufft-cu12==11.3.3.41 nvidia-curand-cu12==10.3.9.55 nvidia-cusolver-cu12==11.7.2.55 nvidia-cusparse-cu12==12.5.4.2 nvidia-cusparselt-cu12==0.6.3 nvidia-nvjitlink-cu12==12.8.61 nvidia-nvtx-cu12==12.8.55 --extra-index-url https://pypi.ngc.nvidia.com
    ```

### macOS

*   **Homebrew:**
    ```shell
    brew install --cask buzz
    ```
*   **.dmg:** Download from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   **.exe:** Download and run the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).  (You may need to bypass the security warning)
*   **winget:**
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

## Screenshots

[Include Screenshots from the original README here]

## Resources

*   [Documentation](https://chidiwilliams.github.io/buzz/)
*   [Buzz Captions on the App Store](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200)
*   [FAQ](https://chidiwilliams.github.io/buzz/docs/faq)