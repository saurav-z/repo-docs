# VibeVoice ComfyUI Nodes: Unleash Realistic Text-to-Speech in ComfyUI

**Bring your creative visions to life with high-fidelity voice synthesis directly within ComfyUI!** This powerful integration provides seamless access to Microsoft's VibeVoice text-to-speech models, enabling you to generate captivating single and multi-speaker audio with ease.

[Original Repo](https://github.com/Enemyx-net/VibeVoice-ComfyUI)

## Key Features

*   🎤 **Single Speaker TTS:** Generate natural-sounding speech with or without voice cloning.
*   👥 **Multi-Speaker Conversations:** Create dynamic dialogues with up to 4 distinct voices.
*   🎙️ **Voice Cloning:** Easily clone voices from audio samples for personalized audio.
*   📝 **Text Input Flexibility:** Load scripts from text files or directly input text.
*   📏 **Automatic Text Chunking:** Handle long texts with configurable chunk sizes.
*   ⏱️ **Custom Pauses:** Control speech pacing with `[pause]` and `[pause:ms]` tags.
*   🔗 **Node Chaining:** Build complex audio workflows by connecting multiple nodes.
*   🛑 **Interruption Support:** Cancel operations at any time.
*   ⚙️ **Model Selection:** Choose from three optimized model variants for different needs.
*   🔊 **Flexible Configuration:** Fine-tune audio with temperature, sampling, and guidance controls.
*   🚀 **Performance Optimization:** Utilize attention mechanisms, diffusion steps, and memory management options.
*   🍎 **Apple Silicon Support:** Native GPU acceleration on M1/M2/M3 Macs.
*   💾 **4-Bit Quantization:** Reduce memory usage without significant quality loss.

## Getting Started

### Automatic Installation (Recommended)

1.  **Navigate** to your ComfyUI custom nodes directory:
    ```bash
    cd ComfyUI/custom_nodes
    ```
2.  **Clone** this repository:
    ```bash
    git clone https://github.com/Enemyx-net/VibeVoice-ComfyUI
    ```
3.  **Restart** ComfyUI. The necessary dependencies will be installed automatically.

## Core Nodes

### 1. VibeVoice Load Text From File

*   **Function:** Loads text from .txt files.
*   **Output:** Text string for TTS nodes.

### 2. VibeVoice Single Speaker

*   **Input:** Text to convert to speech (direct or from Load Text).
*   **Models:** VibeVoice-1.5B, VibeVoice-Large, VibeVoice-Large-Quant-4Bit.
*   **Features:**
    *   Voice cloning from audio input.
    *   Customizable parameters (diffusion steps, temperature, etc.).

### 3. VibeVoice Multiple Speakers

*   **Input:** Text with speaker labels (`[N]:`, 1-4 speakers).
*   **Models:** VibeVoice-1.5B, VibeVoice-Large, VibeVoice-Large-Quant-4Bit.
*   **Features:**
    *   Optional voice samples for each speaker.
    *   Adjustable parameters (diffusion steps, temperature, etc.).

### 4. VibeVoice Free Memory

*   **Function:** Manually frees VibeVoice models from memory.
*   **Use Case:** Optimize memory usage in complex workflows.

## Multi-Speaker Text Format

Use the following format to specify multiple speakers:

```
[1]: Hello, how are you doing today?
[2]: I'm doing great, thanks!
```

*   Speaker labels: `[1]:`, `[2]:`, `[3]:`, `[4]:` (maximum 4 speakers).
*   Optional voice samples for each speaker.

## Model Information

*   **VibeVoice-1.5B:** Fast, good for single speaker. (5GB VRAM)
*   **VibeVoice-Large:** Best quality, for high-quality production and multi-speaker conversations. (17GB VRAM)
*   **VibeVoice-Large-Quant-4Bit:** Good quality, lower VRAM usage (7GB VRAM)

## Generation Modes

*   **Deterministic:** `use_sampling = False` - Consistent, stable output.
*   **Sampling:** `use_sampling = True` - More variation using temperature and top_p.

## Voice Cloning

1.  Connect an audio node to `voice_to_clone` (single speaker) or `speakerX_voice` (multi-speaker).
2.  Requirements: Clear audio with minimal noise, ideally 30 seconds or longer.

## Pause Tag Support

*   **Insert silence in speech.**
*   Use: `[pause]` (1-second) or `[pause:ms]` (custom duration in milliseconds).
*   **Note:** Pauses split text, which may affect context understanding.

## Tips for Best Results

1.  Prepare text with punctuation and formatting.
2.  Select the appropriate model based on your needs.
3.  Save seeds for consistent character voices.
4.  Utilize GPU acceleration for faster generation.

## System Requirements

*   **Minimum:** 8GB VRAM.
*   **Recommended:** 17GB+ VRAM.
*   **Software:** Python 3.8+, PyTorch 2.0+, CUDA 11.8+ (GPU), ComfyUI.

## Troubleshooting & Support

*   Check the troubleshooting section and ComfyUI logs.
*   Ensure VibeVoice is installed correctly.
*   Open an issue with detailed error information.

## Contributing

Contributions are welcome! Follow the guidelines in the original README.

## Changelog

Refer to the original README for the full changelog, highlighting the latest updates, including:

*   Custom pause tag support.
*   Automatic text chunking.
*   SageAttention support.
*   4-bit quantized model support.
*   MPS support for Apple Silicon.
*   Universal Transformers Compatibility.
*   Various bug fixes and improvements.