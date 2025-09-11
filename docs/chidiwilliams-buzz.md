# Buzz: Offline Audio Transcription and Translation Powered by Whisper

Buzz is a powerful, open-source tool that enables you to transcribe and translate audio directly on your computer using the cutting-edge OpenAI Whisper model. ([Original Repo](https://github.com/chidiwilliams/buzz))

**Key Features:**

*   **Offline Functionality:** Transcribe and translate audio without an internet connection, ensuring privacy and convenience.
*   **Powered by Whisper:** Leverages the advanced speech recognition capabilities of OpenAI's Whisper for accurate results.
*   **Cross-Platform Compatibility:** Available for macOS, Windows, and Linux, offering flexibility for all users.
*   **Multiple Installation Options:** Install via PyPI, brew, winget, Flatpak, or Snap for easy setup on your preferred system.
*   **Mac App Store Version:** Explore a native Mac app version with enhanced features like audio playback, transcript editing, and more. [Download on the Mac App Store](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200)

**Installation:**

Choose your preferred method:

*   **PyPI:**

    1.  Install [ffmpeg](https://www.ffmpeg.org/download.html)
    2.  `pip install buzz-captions`
    3.  `python -m buzz`

    **GPU Support for PyPI (Nvidia):**
    ```bash
    pip3 install -U torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
    pip3 install nvidia-cublas-cu12==12.8.3.14 nvidia-cuda-cupti-cu12==12.8.57 nvidia-cuda-nvrtc-cu12==12.8.61 nvidia-cuda-runtime-cu12==12.8.57 nvidia-cudnn-cu12==9.7.1.26 nvidia-cufft-cu12==11.3.3.41 nvidia-curand-cu12==10.3.9.55 nvidia-cusolver-cu12==11.7.2.55 nvidia-cusparse-cu12==12.5.4.2 nvidia-cusparselt-cu12==0.6.3 nvidia-nvjitlink-cu12==12.8.61 nvidia-nvtx-cu12==12.8.55 --extra-index-url https://pypi.ngc.nvidia.com
    ```

*   **macOS:**

    *   **Homebrew:** `brew install --cask buzz`
    *   **Direct Download:** Download the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).
*   **Windows:**

    *   **Direct Download:** Download the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).  You may need to select `More info` -> `Run anyway` if you get a warning.
    *   **Winget:** `winget install ChidiWilliams.Buzz`
*   **Linux:**

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

*   **Latest Development Version:**  See the [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version) for information.

**Screenshots:**

<!-- Screenshots remain the same -->

```
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