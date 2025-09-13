# VibeVoice ComfyUI Nodes: Unleash High-Quality AI Voice Synthesis in Your Workflows

**Elevate your ComfyUI creations with the power of Microsoft's VibeVoice, offering stunningly realistic text-to-speech capabilities with single and multi-speaker support, voice cloning, and advanced control.**

[Link to Original Repo: https://github.com/Enemyx-net/VibeVoice-ComfyUI]

## Key Features

*   🎤 **Single Speaker TTS:** Generate natural speech with optional voice cloning.
*   🗣️ **Multi-Speaker Conversations:** Create dynamic dialogues with up to 4 distinct voices.
*   🔊 **Voice Cloning:** Clone voices from audio samples for personalized audio.
*   📝 **Text File Loading:** Import scripts easily from text files.
*   🧱 **Automatic Text Chunking:** Seamlessly handles long texts for uninterrupted audio.
*   ⏱️ **Custom Pause Tags:** Fine-tune pacing with `[pause]` and `[pause:ms]` tags (wrapper feature).
*   🔗 **Node Chaining:** Connect multiple VibeVoice nodes for intricate workflows.
*   🛑 **Interruption Support:** Cancel operations before or between generations.
*   🚀 **Model Options:** Choose between VibeVoice-1.5B (fastest), VibeVoice-Large (best quality), and VibeVoice-Large-Quant-4Bit (balanced).
*   ⚙️ **Flexible Configuration:** Customize temperature, sampling, guidance scale, and more.
*   ⚡ **Performance Optimization:** Select attention mechanisms, adjust diffusion steps, and manage memory for efficiency.
*   🍎 **Apple Silicon Support:** Harness native GPU acceleration on M1/M2/M3 Macs via MPS.
*   💾 **4-Bit Quantization:** Reduce memory usage with minimal quality loss.
*   📦 **Self-Contained:** No external dependencies, embedded VibeVoice code.
*   💻 **Cross-Platform:** Works on Windows, Linux, and macOS.

## Video Demo

[![VibeVoice ComfyUI Wrapper Demo](https://img.youtube.com/vi/fIBMepIBKhI/maxresdefault.jpg)](https://www.youtube.com/watch?v=fIBMepIBKhI)
**Click to watch the demo video**

## Installation

### Automatic Installation (Recommended)

1.  Navigate to your ComfyUI custom nodes folder:
    ```bash
    cd ComfyUI/custom_nodes
    ```

2.  Clone the VibeVoice ComfyUI repository:
    ```bash
    git clone https://github.com/Enemyx-net/VibeVoice-ComfyUI
    ```

3.  Restart ComfyUI. Required dependencies will automatically install on first use.

## Available Nodes

### 1. VibeVoice Load Text From File

*   **Description:** Loads text content from files in ComfyUI's input/output/temp directories.
*   **Supported Formats:** `.txt`
*   **Output:** Text string for TTS nodes.

### 2. VibeVoice Single Speaker

*   **Description:** Generates speech from text using a single voice.
*   **Text Input:** Direct text or connection from Load Text node.
*   **Models:** VibeVoice-1.5B, VibeVoice-Large, or VibeVoice-Large-Quant-4Bit.
*   **Voice Cloning:** Optional audio input for voice cloning.
*   **Parameters:**
    *   `text`: Input text.
    *   `model`: Select a model.
    *   `attention_type`: auto, eager, sdpa, flash\_attention\_2 or sage (default: auto).
    *   `free_memory_after_generate`: Free VRAM after generation (default: True).
    *   `diffusion_steps`: Number of denoising steps (5-100, default: 20).
    *   `seed`: Random seed (default: 42).
    *   `cfg_scale`: Classifier-free guidance (1.0-2.0, default: 1.3).
    *   `use_sampling`: Enable/disable deterministic generation (default: False).
    *   `voice_to_clone`: Audio input for voice cloning (optional).
    *   `temperature`: Sampling temperature (0.1-2.0, default: 0.95) (optional).
    *   `top_p`: Nucleus sampling parameter (0.1-1.0, default: 0.95) (optional).
    *   `max_words_per_chunk`: Maximum words per chunk for long texts (100-500, default: 250) (optional).

### 3. VibeVoice Multiple Speakers

*   **Description:** Generates multi-speaker conversations with distinct voices.
*   **Speaker Format:** Use `[N]:` notation where N is 1-4.
*   **Voice Assignment:** Optional voice samples for each speaker.
*   **Recommended Model:** VibeVoice-Large for superior multi-speaker quality.
*   **Parameters:**
    *   `text`: Input text with speaker labels.
    *   `model`: Select a model.
    *   `attention_type`: auto, eager, sdpa, flash\_attention\_2 or sage (default: auto).
    *   `free_memory_after_generate`: Free VRAM after generation (default: True).
    *   `diffusion_steps`: Number of denoising steps (5-100, default: 20).
    *   `seed`: Random seed (default: 42).
    *   `cfg_scale`: Classifier-free guidance (1.0-2.0, default: 1.3).
    *   `use_sampling`: Enable/disable deterministic generation (default: False).
    *   `speaker1_voice` to `speaker4_voice`: Audio inputs for voice cloning (optional).
    *   `temperature`: Sampling temperature (0.1-2.0, default: 0.95) (optional).
    *   `top_p`: Nucleus sampling parameter (0.1-1.0, default: 0.95) (optional).

### 4. VibeVoice Free Memory

*   **Description:** Manually frees VibeVoice models from memory.
*   **Input:** `audio` - Connect audio output to trigger memory cleanup.
*   **Output:** `audio` - Passes through the input audio unchanged.
*   **Use Case:** Insert between nodes to free VRAM/RAM at specific points.
*   **Example:** `[VibeVoice Node] → [Free Memory] → [Save Audio]`

## Multi-Speaker Text Format

Format your text with `[N]:` notation for multi-speaker generation:

```
[1]: Hello, how are you today?
[2]: I'm doing great, thanks for asking!
[1]: That's wonderful to hear.
[3]: Hey everyone, mind if I join the conversation?
[2]: Not at all, welcome!
```

*   Use `[1]:`, `[2]:`, `[3]:`, `[4]:` for speaker labels.
*   Supports a maximum of 4 speakers.
*   The system automatically detects the number of speakers from the text.
*   Each speaker can have an optional voice sample for cloning.

## Model Information

### VibeVoice-1.5B

*   **Size:** ~5GB download.
*   **Speed:** Faster inference.
*   **Quality:** Good for single speaker.
*   **Use Case:** Quick prototyping, single voices.

### VibeVoice-Large

*   **Size:** ~17GB download.
*   **Speed:** Slower inference but optimized.
*   **Quality:** Best available quality.
*   **Use Case:** Highest quality production, multi-speaker conversations.
*   **Note:** Latest official release from Microsoft.

### VibeVoice-Large-Quant-4Bit

*   **Size:** ~7GB download.
*   **Speed:** Balanced inference.
*   **Quality:** Good quality.
*   **Use Case:** Good quality production with less VRAM, multi-speaker conversations.
*   **Note:** Quantized by DevParker.

Models are automatically downloaded on first use and cached in `ComfyUI/models/vibevoice/`.

## Generation Modes

### Deterministic Mode (Default)

*   `use_sampling = False`
*   Produces consistent, stable output.
*   Recommended for production.

### Sampling Mode

*   `use_sampling = True`
*   More variation in output.
*   Uses temperature and top\_p parameters.
*   Ideal for creative exploration.

## Voice Cloning

To clone a voice:

1.  Connect an audio node to the `voice_to_clone` input (single speaker) or `speaker1_voice`, `speaker2_voice`, etc. (multi-speaker).
2.  The model will attempt to match the voice characteristics.

**Voice Sample Requirements:**

*   Clear audio with minimal background noise.
*   Minimum 3–10 seconds, with at least 30 seconds recommended.
*   Automatically resampled to 24kHz.

## Pause Tags Support

### Overview

The VibeVoice wrapper includes a custom pause tag feature.  **This is an original implementation of our wrapper** to give you precise control over the speech pacing.

**Available from version 1.3.0**

### Usage

Use two types of pause tags in your text:

*   `[pause]` - Inserts a 1-second silence (default).
*   `[pause:ms]` - Inserts a custom duration silence in milliseconds (e.g., `[pause:2000]` for 2 seconds).

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

⚠️ **Context Limitation Warning**:

> **Note: The pause forces the text to be split into chunks. This may worsen the model's ability to understand the context. The model's context is represented ONLY by its own chunk.**

This means:

*   Text before a pause and text after a pause are processed separately.
*   The model cannot see across pause boundaries when generating speech.
*   This may affect prosody and intonation consistency.
*   Use pauses sparingly for best results.

### How It Works

1.  The wrapper parses your text to find pause tags.
2.  Text segments between pauses are processed independently.
3.  Silence audio is generated for each pause duration.
4.  All audio segments (speech and silence) are concatenated.

### Best Practices

*   Use pauses at natural breaking points (end of sentences, paragraphs).
*   Avoid pauses in the middle of phrases where context is crucial.
*   Test different pause durations for the most natural sound.

## Tips for Best Results

1.  **Text Preparation:**
    *   Use proper punctuation.
    *   Break long texts into paragraphs.
    *   For multi-speaker, ensure clear speaker transitions.
    *   Use pause tags judiciously.
2.  **Model Selection:**
    *   Use 1.5B for quick single-speaker tasks.
    *   Use Large for best quality and multi-speaker.
    *   Use Large-Quant-4Bit for good quality with low VRAM.
3.  **Seed Management:**
    *   Default seed (42) works well.
    *   Save good seeds for consistent character voices.
    *   Try random seeds.
4.  **Performance:**
    *   First run downloads models (5-17GB).
    *   Subsequent runs use cached models.
    *   GPU recommended.

## System Requirements

### Hardware

*   **Minimum:** 8GB VRAM for VibeVoice-1.5B.
*   **Recommended:** 17GB+ VRAM for VibeVoice-Large.
*   **RAM:** 16GB+ system memory.

### Software

*   Python 3.8+.
*   PyTorch 2.0+.
*   CUDA 11.8+ (for GPU acceleration).
*   Transformers 4.51.3+.
*   ComfyUI (latest version).

## Troubleshooting

### Installation Issues

*   Ensure you are using ComfyUI's Python environment.
*   Try manual installation if automatic fails.
*   Restart ComfyUI after installation.

### Generation Issues

*   If voices sound unstable, try deterministic mode.
*   For multi-speaker, verify correct `[N]:` format.
*   Check that speaker numbers are sequential.

### Memory Issues

*   Large models require ~16GB VRAM.
*   Use the 1.5B model for lower VRAM systems.
*   Models use bfloat16 precision.

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

| Model                  | VRAM Usage | Context Length | Max Audio Duration |
|------------------------|------------|----------------|-------------------|
| VibeVoice-1.5B         | ~8GB        | 64K tokens     | ~90 minutes       |
| VibeVoice-Large        | ~17GB       | 32K tokens     | ~45 minutes       |
| VibeVoice-Large-Quant-4Bit | ~7GB        | 32K tokens     | ~45 minutes       |

## Known Limitations

*   Maximum 4 speakers in multi-speaker mode.
*   Works best with English and Chinese text.
*   Some seeds may produce unstable output.
*   Background music generation cannot be directly controlled.

## License

This ComfyUI wrapper is released under the MIT License. See the LICENSE file for details.

**Note**: The VibeVoice model is subject to Microsoft's licensing terms:

*   VibeVoice is for research purposes only.
*   Check Microsoft's VibeVoice repository for full model license details.

## Credits

*   **VibeVoice Model:** Microsoft Research
*   **ComfyUI Integration:** Fabio Sarracino
*   **Base Model:** Built on Qwen2.5 architecture

## Support

For issues or questions:

1.  Check the troubleshooting section.
2.  Review ComfyUI logs.
3.  Ensure VibeVoice is properly installed.
4.  Open an issue with detailed information.

## Contributing

Contributions are welcome! Please:

1.  Test changes thoroughly.
2.  Follow the existing code style.
3.  Update documentation.
4.  Submit pull requests with clear descriptions.

## Changelog

### Version 1.3.0

*   Added custom pause tag support for speech pacing control.
    *   New `[pause]` tag for 1-second silence (default).
    *   New `[pause:ms]` tag for custom duration in milliseconds (e.g., `[pause:2000]` for 2 seconds).
    *   Works with both Single Speaker and Multiple Speakers nodes.
    *   Automatically splits text at pause points.
    *   Note: This is a wrapper feature.

### Version 1.2.5

*   Bug Fixing

### Version 1.2.4

*   Added automatic text chunking for long texts in Single Speaker node.
    *   Single Speaker node now automatically splits texts longer than 250 words.
    *   New optional parameter `max_words_per_chunk` (range: 100-500 words, default: 250).
    *   Maintains consistent voice characteristics.
    *   Seamlessly concatenates audio chunks.

### Version 1.2.3

*   Added SageAttention support for inference speedup.
    *   New attention option "sage" using quantized attention (INT8/FP8).
    *   Requirements: NVIDIA GPU with CUDA and sageattention library installation.

### Version 1.2.2

*   Added 4-bit quantized model support.
    *   New model in menu: `VibeVoice-Large-Quant-4Bit` using ~7GB VRAM instead of ~17GB.
    *   Requirements: NVIDIA GPU with CUDA and bitsandbytes library installed.

### Version 1.2.1

*   Bug Fixing

### Version 1.2.0

*   MPS Support for Apple Silicon.
    *   Added GPU acceleration for Mac with Apple Silicon (M1/M2/M3).
    *   Automatically detects and uses MPS, providing significant performance improvements.

### Version 1.1.1

*   Universal Transformers Compatibility.
    *   Implemented adaptive system to different transformers versions.
    *   Guaranteed compatibility from v4.51.3 onwards.
    *   Auto-detects and adapts to API changes.

### Version 1.1.0

*   Updated the URL for downloading the VibeVoice-Large model
*   Removed VibeVoice-Large-Preview deprecated model

### Version 1.0.9

*   Embedded VibeVoice code directly into the wrapper.
    *   Added vvembed folder containing the complete VibeVoice code (MIT licensed).
    *   No longer requires external VibeVoice installation.
    *   Ensures continued functionality for all users.

### Version 1.0.8

*   BFloat16 Compatibility Fix
    *   Fixed tensor type compatibility issues with audio processing nodes.
    *   Input audio tensors are now converted from BFloat16 to Float32 for numpy compatibility
    *   Output audio tensors are explicitly converted to Float32 to ensure compatibility with downstream nodes
    *   Resolves "Got unsupported ScalarType BFloat16" errors when using voice cloning or saving audio

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

*   Added `attention_type` parameter to both Single Speaker and Multi Speaker nodes for performance optimization.
    *   auto (default): Automatic selection of best implementation
    *   eager: Standard implementation without optimizations
    *   sdpa: PyTorch's optimized Scaled Dot Product Attention
    *   flash_attention_2: Flash Attention 2 for maximum performance (requires compatible GPU)
*   Added `diffusion_steps` parameter to control generation quality vs speed trade-off
    *   Default: 20 (VibeVoice default)
    *   Higher values: Better quality, longer generation time
    *   Lower values: Faster generation, potentially lower quality

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
*   Single speaker node with voice cloning
*   Multi-speaker node with automatic speaker detection
*   Text file loading from ComfyUI directories
*   Deterministic and sampling generation modes
*   Support for VibeVoice 1.5B and Large models