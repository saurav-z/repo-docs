# Buzz: Offline Audio Transcription and Translation Powered by Whisper

**Effortlessly transcribe and translate audio on your computer with Buzz, leveraging the power of OpenAI's Whisper.**

[View the original repository on GitHub](https://github.com/chidiwilliams/buzz)

**Key Features:**

*   **Offline Transcription & Translation:** Process audio without an internet connection using OpenAI's Whisper.
*   **Multi-Platform Support:** Available for macOS, Windows, and Linux (Flatpak & Snap).
*   **Multiple Installation Options:**  Install via PyPI, Homebrew, Winget, Flatpak, or Snap for flexibility.
*   **GPU Acceleration (PyPI):**  Utilize NVIDIA GPUs for faster processing on Windows.
*   **Mac App Store Version:** Experience an enhanced, native macOS version of Buzz on the [Mac App Store](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200) with improved features like audio playback, transcript editing, and drag-and-drop import.
*   **Regular Updates:**  Stay up-to-date with the latest features and bug fixes by exploring the FAQ for the development version.

**Installation:**

Choose your preferred installation method:

### **PyPI**

1.  Install [ffmpeg](https://www.ffmpeg.org/download.html).
2.  Install Buzz:
    ```shell
    pip install buzz-captions
    python -m buzz
    ```
3. **GPU Support for PyPI (Windows)**

```
pip3 install -U torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
pip3 install nvidia-cublas-cu12==12.8.3.14 nvidia-cuda-cupti-cu12==12.8.57 nvidia-cuda-nvrtc-cu12==12.8.61 nvidia-cuda-runtime-cu12==12.8.57 nvidia-cudnn-cu12==9.7.1.26 nvidia-cufft-cu12==11.3.3.41 nvidia-curand-cu12==10.3.9.55 nvidia-cusolver-cu12==11.7.2.55 nvidia-cusparse-cu12==12.5.4.2 nvidia-cusparselt-cu12==0.6.3 nvidia-nvjitlink-cu12==12.8.61 nvidia-nvtx-cu12==12.8.55 --extra-index-url https://pypi.ngc.nvidia.com
```

### **macOS**

*   **Homebrew:**
    ```shell
    brew install --cask buzz
    ```
*   **Direct Download:** Download the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### **Windows**

*   **Direct Download:** Download and run the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest). You may need to select `More info` -> `Run anyway` due to the app not being signed.
*   **Winget:**
    ```shell
    winget install ChidiWilliams.Buzz
    ```

### **Linux**

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

**[Documentation](https://chidiwilliams.github.io/buzz/)**

**[Buzz Captions on the Mac App Store](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200)**

**Screenshots:**

*(Screenshot gallery would be included here)*