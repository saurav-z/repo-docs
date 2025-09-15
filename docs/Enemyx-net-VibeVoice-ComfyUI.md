# VibeVoice ComfyUI Nodes: Unleash Natural-Sounding Speech with Microsoft's Cutting-Edge TTS Models

**Bring the power of Microsoft's VibeVoice text-to-speech technology directly into your ComfyUI workflows, enabling you to create stunning audio with ease!**  [Go to the original repo](https://github.com/Enemyx-net/VibeVoice-ComfyUI)

## Key Features

*   🎤 **Single & Multi-Speaker Synthesis:** Effortlessly generate natural speech with a single voice or up to four distinct speakers.
*   🗣️ **Voice Cloning:** Clone voices from audio samples to create personalized and unique audio outputs.
*   💬 **Multi-Speaker Conversations:** Easily create conversations using `[N]:` notation with the support for distinct voices.
*   📚 **Flexible Input:** Load text from files, direct text input, and integrate with other ComfyUI nodes.
*   💾 **Memory Optimization:** Choose from three VibeVoice model sizes optimized for various hardware configurations. Includes VRAM cleanup and a dedicated "Free Memory" node for manual control.
*   ⚙️ **Customization Options:** Control temperature, sampling, and guidance scale for unique outputs.
*   🍎 **Apple Silicon Support:** Native GPU acceleration on M1/M2/M3 Macs via MPS.
*   ⚡ **Performance Enhancements:** Choose from different attention mechanisms and adjust diffusion steps for an optimal balance of quality and speed.
*   🧩 **Pause Tag Support:** Insert silences and control speech pacing for even more natural-sounding output using `[pause]` and `[pause:ms]` tags.
*   📦 **Self-Contained:** The VibeVoice code is embedded, eliminating external dependencies.

## Getting Started

### Automatic Installation (Recommended)

1.  Navigate to your ComfyUI custom nodes directory:
    ```bash
    cd ComfyUI/custom_nodes
    ```

2.  Clone the VibeVoice ComfyUI repository:
    ```bash
    git clone https://github.com/Enemyx-net/VibeVoice-ComfyUI
    ```

3.  Restart ComfyUI to automatically install the required dependencies.

## Available Nodes

### 1. VibeVoice Load Text From File

*   Loads text from `.txt` files within ComfyUI's input/output/temp directories.
*   **Output:** Text string for TTS nodes.

### 2. VibeVoice Single Speaker

*   Generates speech from input text, supports voice cloning.
*   **Input:** Text (direct or from "Load Text" node).
*   **Models:** VibeVoice-1.5B, VibeVoice-Large, or VibeVoice-Large-Quant-4Bit.
*   **Voice Cloning:** Audio input for voice cloning.
*   **Parameters:**
    *   `text`: Input text.
    *   `model`: Model selection.
    *   `attention_type`: Auto, eager, sdpa, flash_attention_2, or sage.
    *   `free_memory_after_generate`: Frees VRAM after generation (default: True).
    *   `diffusion_steps`: Denoising steps (5-100, default: 20).
    *   `seed`: Random seed (default: 42).
    *   `cfg_scale`: Classifier-free guidance (1.0-2.0, default: 1.3).
    *   `use_sampling`: Enable/disable deterministic generation (default: False).
    *   **Optional Parameters:**
        *   `voice_to_clone`: Audio input for voice cloning.
        *   `temperature`: Sampling temperature (0.1-2.0, default: 0.95).
        *   `top_p`: Nucleus sampling parameter (0.1-1.0, default: 0.95).
        *   `max_words_per_chunk`: Maximum words per chunk for long texts (100-500, default: 250).

### 3. VibeVoice Multiple Speakers

*   Generates multi-speaker conversations.
*   **Input:** Text with speaker labels (e.g., `[1]:`, `[2]:`).
*   **Models:** VibeVoice-1.5B, VibeVoice-Large, or VibeVoice-Large-Quant-4Bit.
*   **Voice Assignment:** Optional voice samples for each speaker.
*   **Parameters:**
    *   `text`: Input text with speaker labels.
    *   `model`: Model selection.
    *   `attention_type`: Auto, eager, sdpa, flash_attention_2, or sage.
    *   `free_memory_after_generate`: Frees VRAM after generation (default: True).
    *   `diffusion_steps`: Denoising steps (5-100, default: 20).
    *   `seed`: Random seed (default: 42).
    *   `cfg_scale`: Classifier-free guidance (1.0-2.0, default: 1.3).
    *   `use_sampling`: Enable/disable deterministic generation (default: False).
        *   **Optional Parameters:**
            *   `speaker1_voice` to `speaker4_voice`: Audio inputs for voice cloning.
            *   `temperature`: Sampling temperature (0.1-2.0, default: 0.95).
            *   `top_p`: Nucleus sampling parameter (0.1-1.0, default: 0.95).

### 4. VibeVoice Free Memory

*   Manually frees loaded VibeVoice models from memory to free VRAM/RAM.
*   **Input:** `audio` (connect audio output).
*   **Output:** `audio` (passes through input audio).
*   **Use Case:** Insert between nodes to free VRAM/RAM at specific workflow points.

## Multi-Speaker Text Format

Use the following format for multi-speaker generation:

```
[1]: Hello, how are you today?
[2]: I'm doing great, thanks!
[1]: That's wonderful.
```

*   Use `[1]:`, `[2]:`, `[3]:`, `[4]:` for speaker labels (up to 4 speakers).
*   The system automatically detects the number of speakers.
*   Each speaker can optionally have a voice sample for cloning.

## Model Information

### VibeVoice-1.5B

*   **Size:** ~5GB download.
*   **Speed:** Fast inference.
*   **Quality:** Good for single-speaker tasks.
*   **Use Case:** Prototyping, single voices.

### VibeVoice-Large

*   **Size:** ~17GB download.
*   **Speed:** Slower inference.
*   **Quality:** Best quality.
*   **Use Case:** High-quality production, multi-speaker conversations.
*   **Note:** Latest official release from Microsoft.

### VibeVoice-Large-Quant-4Bit

*   **Size:** ~7GB download.
*   **Speed:** Balanced inference.
*   **Quality:** Good quality.
*   **Use Case:** Good quality production with less VRAM, multi-speaker conversations.
*   **Note:** Quantized by DevParker.

Models are automatically downloaded and cached in `ComfyUI/models/vibevoice/`.

## Generation Modes

### Deterministic Mode (Default)

*   `use_sampling = False`.
*   Consistent output.
*   Recommended for production.

### Sampling Mode

*   `use_sampling = True`.
*   More variation in output.
*   Uses `temperature` and `top_p` parameters.
*   Good for creative exploration.

## Voice Cloning

To clone a voice:

1.  Connect an audio node to the `voice_to_clone` input (single speaker).
2.  Or, connect to `speaker1_voice`, `speaker2_voice`, etc. (multi-speaker).
3.  The model will attempt to match the voice characteristics.

**Requirements for voice samples:**

*   Clear audio with minimal background noise.
*   Minimum 3-10 seconds (30 seconds recommended).
*   Automatically resampled to 24kHz.

## Pause Tags Support

### Overview

Insert silences for speech pacing control!

**Available from version 1.3.0**

### Usage

*   `[pause]` - 1-second silence.
*   `[pause:ms]` - Custom duration in milliseconds (e.g., `[pause:2000]` for 2 seconds).

### Examples

#### Single Speaker

```
Welcome to our presentation. [pause] Today we'll explore AI. [pause:500] Let's begin!
```

#### Multi-Speaker

```
[1]: Hello everyone [pause] how are you doing today?
[2]: I'm doing great! [pause:500] Thanks for asking.
[1]: Wonderful to hear!
```

### Important Notes

⚠️ **Context Limitation Warning**:

*   Text before and after pauses is processed separately.
*   Affects prosody and intonation.
*   Use pauses sparingly for best results.

### How It Works

1.  Wrapper parses text for pause tags.
2.  Text segments between pauses are processed independently.
3.  Silence is generated for each pause duration.
4.  Segments are concatenated.

### Best Practices

*   Use pauses at natural breaks.
*   Avoid pauses in the middle of phrases.
*   Test different durations.

## Tips for Best Results

1.  **Text Preparation:** Use proper punctuation, break up long texts, and ensure clear speaker transitions.
2.  **Model Selection:** Use 1.5B for speed, Large for quality, and Large-Quant-4Bit for VRAM efficiency.
3.  **Seed Management:** Use default seed (42) or save/try random seeds for consistent voices.
4.  **Performance:** GPU recommended; models are cached.

## System Requirements

### Hardware

*   **Minimum:** 8GB VRAM (VibeVoice-1.5B)
*   **Recommended:** 17GB+ VRAM (VibeVoice-Large)
*   **RAM:** 16GB+ system memory.

### Software

*   Python 3.8+
*   PyTorch 2.0+
*   CUDA 11.8+ (for GPU acceleration)
*   Transformers 4.51.3+
*   ComfyUI (latest version)

## Troubleshooting

### Installation Issues

*   Ensure you're using ComfyUI's environment.
*   Try manual installation if automatic fails.
*   Restart ComfyUI after installation.

### Generation Issues

*   Try deterministic mode for unstable voices.
*   Check the `[N]:` format for multi-speaker.

### Memory Issues

*   Use a smaller model or the Free Memory node.

## Examples

### Single Speaker

```
Text: "Welcome. Today we'll explore AI."
Model: VibeVoice-1.5B
cfg_scale: 1.3
use_sampling: False
```

### Two Speakers

```
[1]: AI developments?
[2]: Yes, impressive!
[1]: Voice synthesis has come far.
[2]: Absolutely.
```

### Four Speaker Conversation

```
[1]: Welcome everyone.
[2]: Thanks for coming!
[3]: Glad to be here.
[4]: Discussion.
[1]: Let's begin.
```

## Performance Benchmarks

| Model                  | VRAM Usage | Context Length | Max Audio Duration |
| :--------------------- | :--------- | :------------- | :----------------- |
| VibeVoice-1.5B         | ~8GB       | 64K tokens     | ~90 minutes        |
| VibeVoice-Large        | ~17GB      | 32K tokens     | ~45 minutes        |
| VibeVoice-Large-Quant-4Bit | ~7GB       | 32K tokens     | ~45 minutes        |

## Known Limitations

*   Maximum 4 speakers in multi-speaker mode.
*   Best with English and Chinese text.
*   Some seeds may produce unstable output.
*   Background music generation is not directly controllable.

## License

This ComfyUI wrapper is released under the MIT License. See the LICENSE file for details.

**Note:** The VibeVoice model itself is subject to Microsoft's licensing terms (research purposes only - check Microsoft's VibeVoice repository for details).

## Credits

*   **VibeVoice Model:** Microsoft Research
*   **ComfyUI Integration:** Fabio Sarracino
*   **Base Model:** Qwen2.5 architecture

## Support

For issues or questions:

1.  Check the troubleshooting section.
2.  Review ComfyUI logs.
3.  Ensure VibeVoice is properly installed.
4.  Open an issue with detailed error information.

## Contributing

Contributions are welcome! Please:

1.  Test changes thoroughly.
2.  Follow the existing code style.
3.  Update the documentation as needed.
4.  Submit pull requests with clear descriptions.

## Changelog

### Version 1.3.0

*   Added custom pause tag support for speech pacing control.

### Version 1.2.5

*   Bug Fixing

### Version 1.2.4

*   Added automatic text chunking for long texts in Single Speaker node

### Version 1.2.3

*   Added SageAttention support for inference speedup

### Version 1.2.2

*   Added 4-bit quantized model support

### Version 1.2.1

*   Bug Fixing

### Version 1.2.0

*   MPS Support for Apple Silicon

### Version 1.1.1

*   Universal Transformers Compatibility

### Version 1.1.0

*   Updated the URL for downloading the VibeVoice-Large model
*   Removed VibeVoice-Large-Preview deprecated model

### Version 1.0.9

*   Embedded VibeVoice code directly into the wrapper

### Version 1.0.8

*   BFloat16 Compatibility Fix

### Version 1.0.7

*   Added interruption handler to detect user's cancel request
*   Bug fixing

### Version 1.0.6

*   Fixed a bug that prevented VibeVoice nodes from receiving audio directly from another VibeVoice node

### Version 1.0.5

*   Added support for Microsoft's official VibeVoice-Large model (stable release)

### Version 1.0.4

*   Improved tokenizer dependency handling

### Version 1.0.3

*   Added `attention_type` parameter to both Single Speaker and Multi Speaker nodes for performance optimization
*   Added `diffusion_steps` parameter to control generation quality vs speed trade-off

### Version 1.0.2

*   Added `free_memory_after_generate` toggle to both Single Speaker and Multi Speaker nodes
*   New dedicated "Free Memory Node" for manual memory management in workflows
*   Improved VRAM/RAM usage optimization
*   Enhanced stability for long generation sessions
*   Users can now choose between automatic or manual memory management

### Version 1.0.1

*   Fixed issue with line breaks in speaker text (both single and multi-speaker nodes)
*   Line breaks within individual speaker text are now automatically removed before generation
*   Improved text formatting handling for all generation modes

### Version 1.0.0

*   Initial release