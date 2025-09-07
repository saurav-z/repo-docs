# Buzz: Offline Audio Transcription & Translation

**Effortlessly transcribe and translate audio files offline using the power of OpenAI's Whisper with Buzz!**  ([Original Repository](https://github.com/chidiwilliams/buzz))

Buzz is a powerful, open-source application that utilizes OpenAI's Whisper to provide accurate and efficient audio transcription and translation directly on your computer.  Enjoy the privacy and speed of offline processing.

## Key Features

*   **Offline Transcription & Translation:** Transcribe and translate audio files without an internet connection, ensuring privacy and speed.
*   **Powered by Whisper:** Leverages the advanced capabilities of OpenAI's Whisper for high-quality results.
*   **Cross-Platform:** Available for macOS, Windows, and Linux.
*   **Multiple Installation Options:** Install via PyPI, macOS (Homebrew or .dmg), Windows (.exe or Winget), or Linux (Flatpak or Snap).
*   **GPU Support:** Optimized for NVIDIA GPUs on Windows (PyPI install) for faster processing.
*   **User-Friendly Interface:** Intuitive design for easy audio import and transcript management.
*   **[Optional] Mac App Store Version:** A dedicated Mac version is available with enhanced features: audio playback, drag-and-drop import, transcript editing, search, and more.  [Get it on the Mac App Store](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200)

## Installation

Choose your preferred platform below:

### macOS

*   **Homebrew:**
    ```bash
    brew install --cask buzz
    ```
*   **.dmg (Manual Download):** Download the .dmg from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   **.exe (Manual Download):** Download and run the .exe from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).  You may need to bypass a security warning ("More info" -> "Run anyway").
*   **Winget:**
    ```bash
    winget install ChidiWilliams.Buzz
    ```
*   **PyPI (with NVIDIA GPU support):**

    1.  Install [ffmpeg](https://www.ffmpeg.org/download.html)
    2.  ```bash
        pip install buzz-captions
        python -m buzz
        ```
    3.  Ensure CUDA support for [torch](https://pytorch.org/get-started/locally/) :
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

## Documentation & Support

*   [Documentation](https://chidiwilliams.github.io/buzz/)
*   For the latest development version, refer to the [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version).

## Screenshots

*(Screenshots remain the same)*

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
```

Key improvements and SEO considerations:

*   **Clear Headline:** "Buzz: Offline Audio Transcription & Translation" is more descriptive and includes relevant keywords.
*   **Hook:** The one-sentence hook immediately highlights the key benefits.
*   **Keyword Optimization:**  Includes keywords like "offline audio transcription," "audio translation," "OpenAI Whisper," and platform names.
*   **Bulleted Key Features:**  Highlights core functionalities in an easy-to-read format.
*   **Platform-Specific Sections:** Makes it easier for users to find installation instructions.
*   **Call to Action (Mac App Store):**  Promotes the Mac App Store version with a clear call to action.
*   **Concise and Readable:** Improves the flow and readability of the original README.
*   **Emphasis on Offline Capabilities:** Reinforces the privacy and speed advantages.
*   **Stronger SEO with More Keywords:** By incorporating more keywords within the text the user can be more likely to find the app.
*   **Links:** Added links back to the repo.
*   **Removed redudant Chinese language notice.**