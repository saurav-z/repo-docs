# VibeVoice ComfyUI: Effortlessly Generate High-Quality Text-to-Speech in ComfyUI

Bring the power of Microsoft's VibeVoice directly into your ComfyUI workflows with this comprehensive and user-friendly integration.  Explore cutting-edge voice synthesis features like single and multi-speaker generation, voice cloning, and advanced control over your audio output.  Get started now by visiting the [original repository](https://github.com/Enemyx-net/VibeVoice-ComfyUI) for the latest updates and more information.

## Key Features

*   🎤 **Single Speaker TTS:** Generate natural-sounding speech with optional voice cloning capabilities.
*   👥 **Multi-Speaker Conversations:** Create dynamic dialogues with support for up to 4 distinct voices.
*   🔊 **Voice Cloning:** Easily clone voices from audio samples to personalize your creations.
*   📝 **Text File Loading:** Import scripts effortlessly from text files directly into your workflow.
*   ⚙️ **Flexible Configuration:** Fine-tune your output with controls for temperature, sampling, and guidance scale.
*   🚀 **Model Options:** Choose from three optimized VibeVoice model variants for performance and quality: VibeVoice-1.5B, VibeVoice-Large, and VibeVoice-Large-Quant-4Bit.
*   ⚡ **Performance Optimization:** Select attention mechanisms, adjust diffusion steps, and manage memory for optimal performance on your hardware.
*   🍎 **Apple Silicon Support:** Native GPU acceleration on M1/M2/M3 Macs via MPS for faster generation.
*   💾 **4-Bit Quantization:** Reduce memory usage while maintaining high-quality audio with the VibeVoice-Large-Quant-4Bit model.
*   ⏸️ **Custom Pause Tags:** Insert silences into your generated speech with custom duration control.

## Getting Started

### Installation

1.  **Automatic Installation (Recommended):**
    *   Navigate to your ComfyUI custom nodes directory: `cd ComfyUI/custom_nodes`
    *   Clone the repository: `git clone https://github.com/Enemyx-net/VibeVoice-ComfyUI`
    *   Restart ComfyUI: The nodes will automatically install the necessary requirements upon first use.

## Available Nodes

### 1. VibeVoice Load Text From File

*   **Description:** Loads text content from files, such as .txt files, located in ComfyUI's input, output, or temp directories.
*   **Output:** Text string for subsequent TTS nodes.

### 2. VibeVoice Single Speaker

*   **Description:** Transforms text into speech using a single voice.
*   **Input:** Direct text or text sourced from the Load Text node.
*   **Models:** VibeVoice-1.5B, VibeVoice-Large, or VibeVoice-Large-Quant-4Bit.
*   **Voice Cloning:** Supports audio input for cloning voices.
*   **Parameters:**
    *   `text`: The input text to convert to speech.
    *   `model`: Select from VibeVoice-1.5B, VibeVoice-Large, or VibeVoice-Large-Quant-4Bit.
    *   `attention_type`: Choose your attention type (auto, eager, sdpa, flash\_attention\_2 or sage)
    *   `free_memory_after_generate`: Frees VRAM after generation (default: True).
    *   `diffusion_steps`: Controls the number of denoising steps (5-100, default: 20).
    *   `seed`: Sets the random seed for reproducible results (default: 42).
    *   `cfg_scale`: Guidance scale for classifier-free generation (1.0-2.0, default: 1.3).
    *   `use_sampling`: Enables or disables deterministic generation (default: False).
*   **Optional Parameters:**
    *   `voice_to_clone`: Input an audio file for voice cloning.
    *   `temperature`: Adjust the sampling temperature (0.1-2.0, default: 0.95).
    *   `top_p`: Use the nucleus sampling parameter (0.1-1.0, default: 0.95).
    *   `max_words_per_chunk`: Sets the maximum number of words per chunk (100-500, default: 250).

### 3. VibeVoice Multiple Speakers

*   **Description:** Enables multi-speaker conversations with distinct voices.
*   **Speaker Format:** Use the `[N]:` notation, where N is a number between 1 and 4 to specify your speaker.
*   **Voice Assignment:** Supports audio inputs for cloning voices for each speaker.
*   **Recommended Model:** VibeVoice-Large for higher-quality multi-speaker results.
*   **Parameters:**
    *   `text`: The input text with speaker labels.
    *   `model`: Select from VibeVoice-1.5B, VibeVoice-Large, or VibeVoice-Large-Quant-4Bit.
    *   `attention_type`: Choose your attention type (auto, eager, sdpa, flash\_attention\_2 or sage)
    *   `free_memory_after_generate`: Frees VRAM after generation (default: True).
    *   `diffusion_steps`: Adjust the number of denoising steps (5-100, default: 20).
    *   `seed`: Sets the random seed for reproducible results (default: 42).
    *   `cfg_scale`: Sets guidance scale for classifier-free generation (1.0-2.0, default: 1.3).
    *   `use_sampling`: Enables or disables deterministic generation (default: False).
*   **Optional Parameters:**
    *   `speaker1_voice` to `speaker4_voice`: Input audio files for cloning voices for each speaker.
    *   `temperature`: Adjust the sampling temperature (0.1-2.0, default: 0.95).
    *   `top_p`: Sets the nucleus sampling parameter (0.1-1.0, default: 0.95).

### 4. VibeVoice Free Memory

*   **Description:** Manually releases loaded VibeVoice models from memory.
*   **Input:** `audio` - Connect the audio output to trigger the memory cleanup.
*   **Output:** `audio` - Passes the audio input through unchanged.
*   **Use Case:** Insert between nodes to free VRAM/RAM at specific workflow points.
*   **Example:** `[VibeVoice Node] → [Free Memory] → [Save Audio]`

## Multi-Speaker Text Format

Structure your text using the `[N]:` notation to create multi-speaker dialogues:

```
[1]: Hello, how are you today?
[2]: I'm doing great, thanks for asking!
[1]: That's wonderful to hear.
[3]: Hey everyone, mind if I join the conversation?
[2]: Not at all, welcome!
```

**Important Considerations:**

*   Use `[1]:`, `[2]:`, `[3]:`, and `[4]:` to label speakers.
*   A maximum of 4 speakers are supported.
*   The system automatically detects the number of speakers.
*   Each speaker can have an optional voice sample for cloning.

## Model Information

### VibeVoice-1.5B

*   **Size:** Approximately 5GB download.
*   **Speed:** Fast inference.
*   **Quality:** Suitable for single speakers.
*   **Use Case:** Ideal for quick prototyping and single-voice projects.

### VibeVoice-Large

*   **Size:** Approximately 17GB download.
*   **Speed:** Slower inference, but optimized.
*   **Quality:** Provides the best available quality.
*   **Use Case:** Best for high-quality productions and multi-speaker conversations.
*   **Note:** This is the latest official release from Microsoft.

### VibeVoice-Large-Quant-4Bit

*   **Size:** Approximately 7GB download.
*   **Speed:** Balanced inference.
*   **Quality:** Good quality.
*   **Use Case:** Suitable for good quality production while conserving VRAM, including multi-speaker conversations.
*   **Note:** Quantized by DevParker.

Models are automatically downloaded on first use and are cached in the `ComfyUI/models/vibevoice/` directory.

## Generation Modes

### Deterministic Mode (Default)

*   `use_sampling = False`
*   Guarantees consistent, stable output.
*   Recommended for production applications.

### Sampling Mode

*   `use_sampling = True`
*   Offers more variation in the output.
*   Uses the `temperature` and `top_p` parameters.
*   Suited for creative exploration.

## Voice Cloning

To clone a voice:

1.  Connect an audio node to the `voice_to_clone` input (for single-speaker generation).
2.  Or, connect audio to `speaker1_voice`, `speaker2_voice`, etc. (for multi-speaker generation).
3.  The model will attempt to match the voice characteristics.

**Requirements for Voice Samples:**

*   Clear audio with minimal background noise is essential.
*   Provide a minimum of 3–10 seconds of audio. It is recommended to provide at least 30 seconds for better quality.
*   Audio samples are automatically resampled to 24kHz.

## Pause Tags Support

### Overview

The VibeVoice wrapper includes a custom pause tag feature to enable more control over the pacing of your generated speech. **This feature is not a standard Microsoft VibeVoice function**. It's an original implementation that allows you to insert silences between text segments to better control speech pacing.

**Available from version 1.3.0**

### Usage

Use two types of pause tags in your text to insert silences:

*   `[pause]` - Inserts a 1-second silence (default).
*   `[pause:ms]` - Inserts a custom duration in milliseconds (e.g., `[pause:2000]` for 2 seconds).

### Examples

#### Single Speaker

```
Welcome to our presentation. [pause] Today we'll explore artificial intelligence. [pause:500] Let's begin!
```

#### Multi-Speaker

```
[1]: Hello everyone [pause] how are you doing today?
[2]: I'm doing great! [pause:500] Thanks for asking.
[1]: Wonderful to hear!
```

### Important Notes

⚠️ **Context Limitation Warning:**

> **Note: The pause forces the text to be split into chunks. This may worsen the model's ability to understand the context. The model's context is represented ONLY by its own chunk.**

This means:

*   Text before a pause and text after a pause are processed separately.
*   The model cannot see across pause boundaries when generating speech.
*   This may affect the prosody and intonation consistency.
*   Use pauses sparingly for the best results.

### How It Works

1.  The wrapper parses your text to find pause tags.
2.  Text segments between pauses are processed independently.
3.  Silence audio is generated for each pause duration.
4.  All audio segments (speech and silence) are concatenated.

### Best Practices

*   Use pauses at natural breaking points (end of sentences, paragraphs).
*   Avoid pauses in the middle of phrases where context is important.
*   Test different pause durations to find what sounds the most natural.

## Tips for Best Results

1.  **Text Preparation:**
    *   Use appropriate punctuation for natural pauses.
    *   Break longer texts into paragraphs.
    *   For multi-speaker dialogues, ensure clear speaker transitions.
    *   Utilize pause tags sparingly to preserve context continuity.

2.  **Model Selection:**
    *   Use the 1.5B model for quicker single-speaker tasks (fastest, requires ~8GB VRAM).
    *   Use the Large model for the best quality and multi-speaker support (~16GB VRAM).
    *   Use the Large-Quant-4Bit model for good quality and to reduce VRAM usage (~7GB VRAM).

3.  **Seed Management:**
    *   The default seed (42) often yields excellent results.
    *   Save the seeds that produce the best voice characteristics for consistency.
    *   Experiment with random seeds if the default seed does not produce satisfactory results.

4.  **Performance:**
    *   Initial runs require downloading models (ranging from 5-17GB).
    *   Subsequent runs will utilize cached models.
    *   A GPU is recommended for faster inference.

## System Requirements

### Hardware

*   **Minimum:** 8GB VRAM (for VibeVoice-1.5B)
*   **Recommended:** 17GB+ VRAM (for VibeVoice-Large)
*   **RAM:** 16GB+ system memory

### Software

*   Python 3.8+
*   PyTorch 2.0+
*   CUDA 11.8+ (for GPU acceleration)
*   Transformers 4.51.3+
*   ComfyUI (latest version)

## Troubleshooting

### Installation Issues

*   Verify that you are using ComfyUI's designated Python environment.
*   Attempt a manual installation if the automatic method fails.
*   Restart ComfyUI after the installation process is complete.

### Generation Issues

*   If the voices sound unstable, try using deterministic mode.
*   For multi-speaker conversations, confirm the correct format, using `[N]:`.
*   Ensure speaker numbers are sequential (e.g., 1, 2, 3, not 1, 3, 5).

### Memory Issues

*   The Large model requires approximately 16GB VRAM.
*   Use the 1.5B model for systems with lower VRAM.
*   Models utilize bfloat16 precision to optimize memory usage.

## Examples

### Single Speaker

```
Text: "Welcome to our presentation. Today we'll explore the fascinating world of artificial intelligence."
Model: VibeVoice-1.5B
cfg_scale: 1.3
use_sampling: False
```

### Two Speakers

```
[1]: Have you seen the new AI developments?
[2]: Yes, they're quite impressive!
[1]: I think voice synthesis has come a long way.
[2]: Absolutely, it sounds so natural now.
```

### Four Speaker Conversation

```
[1]: Welcome everyone to our meeting.
[2]: Thanks for having us!
[3]: Glad to be here.
[4]: Looking forward to the discussion.
[1]: Let's begin with the agenda.
```

## Performance Benchmarks

| Model                    | VRAM Usage | Context Length | Max Audio Duration |
| ------------------------- | ---------- | -------------- | ------------------ |
| VibeVoice-1.5B            | ~8GB       | 64K tokens     | ~90 minutes        |
| VibeVoice-Large           | ~17GB      | 32K tokens     | ~45 minutes        |
| VibeVoice-Large-Quant-4Bit | ~7GB       | 32K tokens     | ~45 minutes        |

## Known Limitations

*   Supports a maximum of 4 speakers in multi-speaker mode.
*   Optimal results with English and Chinese text.
*   Some seeds may result in unstable output.
*   The generation of background music cannot be controlled directly.

## License

This ComfyUI wrapper is released under the MIT License. Please consult the LICENSE file for specific details.

**Note:** The VibeVoice model itself is subject to Microsoft's licensing terms:

*   VibeVoice is intended for research purposes only.
*   Refer to Microsoft's VibeVoice repository for comprehensive details on the model license.

## Links

*   [Original VibeVoice Repository](https://github.com/microsoft/VibeVoice) - Official Microsoft VibeVoice repository (currently unavailable)

## Credits

*   **VibeVoice Model:** Microsoft Research
*   **ComfyUI Integration:** Fabio Sarracino
*   **Base Model:** Built on Qwen2.5 architecture

## Support

For any issues or questions, please:

1.  Review the Troubleshooting section above.
2.  Examine the ComfyUI logs for any error messages.
3.  Ensure that VibeVoice is correctly installed.
4.  If the problem persists, open an issue with detailed error information.

## Contributing

Contributions are welcome! Please:

1.  Thoroughly test any changes.
2.  Adhere to the existing code style.
3.  Update the documentation as needed.
4.  Submit pull requests with clear and comprehensive descriptions.

## Changelog

### Version 1.3.0

*   Added custom pause tag support for speech pacing control.
    *   Added the new `[pause]` tag to insert a 1-second silence (default).
    *   Added the new `[pause:ms]` tag to insert a custom duration in milliseconds (e.g., `[pause:2000]` for 2 seconds).
    *   Works with both Single Speaker and Multiple Speakers nodes.
    *   Automatically splits text at pause points while maintaining voice consistency.
    *   Note: This feature is part of the wrapper and not a standard Microsoft VibeVoice feature.

### Version 1.2.5

*   Bug Fixing

### Version 1.2.4

*   Added automatic text chunking for long texts in the Single Speaker node.
    *   The Single Speaker node automatically splits texts longer than 250 words to prevent audio acceleration issues.
    *   Added a new optional parameter `max_words_per_chunk` (range: 100-500 words, default: 250).
    *   Maintains consistent voice characteristics across all chunks using the same seed.
    *   Seamlessly concatenates audio chunks for smooth, natural output.

### Version 1.2.3

*   Added SageAttention support for inference speedup.
    *   New attention option "sage" using quantized attention (INT8/FP8) for faster generation.
    *   Requires: NVIDIA GPU with CUDA and the sageattention library installed.

### Version 1.2.2

*   Added 4-bit quantized model support.
    *   New model in menu: `VibeVoice-Large-Quant-4Bit` using ~7GB VRAM instead of ~17GB.
    *   Requires: NVIDIA GPU with CUDA and the bitsandbytes library installed.

### Version 1.2.1

*   Bug Fixing

### Version 1.2.0

*   MPS Support for Apple Silicon:
    *   Added GPU acceleration support for Mac with Apple Silicon (M1/M2/M3).
    *   Automatically detects and uses the MPS backend when available, providing significant performance improvements over CPU.

### Version 1.1.1

*   Universal Transformers Compatibility:
    *   Implemented an adaptive system that automatically adjusts to different transformers versions.
    *   Guaranteed compatibility from v4.51.3 onwards.
    *   Auto-detects and adapts to API changes between versions.

### Version 1.1.0

*   Updated the URL for downloading the VibeVoice-Large model.
*   Removed the VibeVoice-Large-Preview deprecated model.

### Version 1.0.9

*   Embedded VibeVoice code directly into the wrapper.
    *   Added a `vvembed` folder containing the complete VibeVoice code (MIT licensed).
    *   No longer requires external VibeVoice installation.
    *   Ensures continued functionality for all users.

### Version 1.0.8

*   BFloat16 Compatibility Fix.
    *   Fixed tensor type compatibility issues with audio processing nodes.
    *   Input audio tensors are now converted from BFloat16 to Float32 for numpy compatibility.
    *   Output audio tensors are explicitly converted to Float32 to ensure compatibility with downstream nodes.
    *   Resolves "Got unsupported ScalarType BFloat16" errors when using voice cloning or saving audio.

### Version 1.0.7

*   Added an interruption handler to detect the user's cancel request.
*   Bug fixing

### Version 1.0.6

*   Fixed a bug that prevented VibeVoice nodes from receiving audio directly from another VibeVoice node.

### Version 1.0.5

*   Added support for Microsoft's official VibeVoice-Large model (stable release).

### Version 1.0.4

*   Improved tokenizer dependency handling.

### Version 1.0.3

*   Added the `attention_type` parameter to both Single Speaker and Multi Speaker nodes for performance optimization.
    *   `auto` (default): Automatic selection of the best implementation.
    *   `eager`: Standard implementation without optimizations.
    *   `sdpa`: PyTorch's optimized Scaled Dot Product Attention.
    *   `flash_attention_2`: Flash Attention 2 for maximum performance (requires a compatible GPU).
*   Added the `diffusion_steps` parameter to control the generation quality vs. speed trade-off.
    *   Default: 20 (VibeVoice default).
    *   Higher values: Better quality, longer generation time.
    *   Lower values: Faster generation, potentially lower quality.

### Version 1.0.2

*   Added the `free_memory_after_generate` toggle to both Single Speaker and Multi Speaker nodes.
*   Added a new dedicated "Free Memory Node" for manual memory management in workflows.
*   Improved VRAM/RAM usage optimization.
*   Enhanced stability for long generation sessions.
*   Users can now choose between automatic or manual memory management.

### Version 1.0.1

*   Fixed an issue with line breaks in speaker text (both single and multi-speaker nodes).
*   Line breaks within individual speaker text are now automatically removed before generation.
*   Improved text formatting handling for all generation modes.

### Version 1.0.0

*   Initial release
    *   Single speaker node with voice cloning.
    *   Multi-speaker node with automatic speaker detection.
    *   Text file loading from ComfyUI directories.
    *   Deterministic and sampling generation modes.
    *   Support for VibeVoice 1.5B and Large models.