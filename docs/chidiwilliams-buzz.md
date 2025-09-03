# Buzz: Offline Audio Transcription & Translation Powered by Whisper

**Effortlessly transcribe and translate audio files locally on your computer with Buzz, leveraging the power of OpenAI's Whisper.**  [View the original repository](https://github.com/chidiwilliams/buzz).

**Key Features:**

*   **Offline Transcription:** Transcribe audio without an internet connection, ensuring privacy and speed.
*   **Multi-Platform Support:** Available for macOS, Windows, and Linux (Flatpak and Snap).
*   **GPU Acceleration:** Take advantage of your NVIDIA GPU for faster transcription using PyTorch.
*   **Translation Capabilities:** Translate audio into various languages.
*   **User-Friendly Interface:**  A clean and intuitive interface for easy audio processing.
*   **Mac App Store Version:** Experience a more refined Buzz with additional features like audio playback, transcript editing, and search.

**Get the Mac-native version with enhanced features:**
<a href="https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&amp;itsct=apps_box_badge&amp;itscg=30200"><img src="https://toolbox.marketingtools.apple.com/api/badges/download-on-the-mac-app-store/black/en-us?size=250x83&amp;releaseDate=1679529600" alt="Download on the Mac App Store" /></a>

![Buzz](./buzz/assets/buzz-banner.jpg)

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

*   **Homebrew:**

    ```shell
    brew install --cask buzz
    ```
*   **Direct Download:** Download the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   **Direct Download:** Download and run the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest). Note that the app is not signed, so you may need to select `More info` -> `Run anyway` during installation.
*   **Winget:**

    ```shell
    winget install ChidiWilliams.Buzz
    ```

*   **GPU Support for PyPI (NVIDIA):**

    To enable GPU acceleration for NVIDIA GPUs, install the appropriate CUDA dependencies after installing the PyPI version:
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

For the most up-to-date features and bug fixes, refer to the [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version).

### Documentation

*   [Documentation](https://chidiwilliams.github.io/buzz/)

### Screenshots

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
Key improvements and SEO considerations:

*   **Clear, Concise Title:** Uses the primary keyword ("Buzz") and target functionality ("Offline Audio Transcription & Translation").
*   **One-Sentence Hook:** Immediately grabs the user's attention and highlights the core benefit.
*   **Keyword Optimization:**  Includes relevant keywords like "offline transcription," "audio translation," "Whisper," "macOS," "Windows," "Linux," and "GPU."
*   **Structured Headings:**  Uses clear headings for readability and SEO ranking.
*   **Bulleted Key Features:** Makes the key benefits easily scannable.
*   **Installation Section Improvement:** Instructions are more clear and grouped. Includes more install options to cover more users.
*   **Call to Action (CTA):** Encourages users to explore the Mac App Store version.
*   **Clear Links:** Provides links to the documentation, original repository, and Mac App Store listing.
*   **Image Optimization:** Added `alt` text to images for accessibility and SEO.
*   **Concise Language:** Removes unnecessary phrases to make the content more focused.
*   **Updated Formatting:** Improved overall readability and visual appeal.