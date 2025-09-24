# VibeVoice ComfyUI Nodes: Transform Text into Natural-Sounding Speech

Unleash the power of Microsoft's VibeVoice directly within your ComfyUI workflows to generate high-quality text-to-speech (TTS) for single and multi-speaker applications.  [Visit the original repository](https://github.com/Enemyx-net/VibeVoice-ComfyUI) for the most up-to-date information and to contribute.

## Key Features

*   🎤 **Single Speaker TTS:** Generate natural speech and optional voice cloning.
*   👥 **Multi-Speaker Conversations:** Create realistic conversations for up to 4 distinct speakers.
*   🗣️ **Voice Cloning:** Clone voices from audio samples for personalized speech.
*   📝 **Text File Loading:** Import scripts directly from text files.
*   🧩 **Automatic Text Chunking:** Handles long texts efficiently with configurable chunk size.
*   ⏱️ **Custom Pause Tags:** Insert silences with  `[pause]` and `[pause:ms]` tags (wrapper feature).
*   🔄 **Node Chaining:** Connect VibeVoice nodes to create complex workflows.
*   🛑 **Interruption Support:** Cancel operations during generation.
*   🚀 **Model Selection:** Choose between optimized model variants for performance or quality.
*   ⚙️ **Flexible Configuration:** Control temperature, sampling, and guidance scale.
*   ⚡ **Performance Optimization:** Select between multiple attention mechanisms to improve generation speed.
*   💾 **Memory Management:** Free VRAM automatically, or use a manual memory control node.
*   🍎 **Apple Silicon Support:** Native GPU acceleration for M1/M2/M3 Macs.
*   🔢 **4-Bit Quantization:**  Reduce memory usage with minimal quality loss.

## Installation

### Automatic Installation (Recommended)

1.  Navigate to your ComfyUI custom nodes directory:
    ```bash
    cd ComfyUI/custom_nodes
    ```
2.  Clone the repository:
    ```bash
    git clone https://github.com/Enemyx-net/VibeVoice-ComfyUI
    ```

3.  Restart ComfyUI; requirements will install automatically on first use.

## Available Nodes

### 1. VibeVoice Load Text From File

*   Loads text from `.txt` files.
*   **Output:** Text string for TTS nodes.

### 2. VibeVoice Single Speaker

*   Generates speech from text using a single voice.
*   **Parameters:**
    *   `text`: Input text.
    *   `model`: VibeVoice-1.5B, VibeVoice-Large, or VibeVoice-Large-Quant-4Bit.
    *   `attention_type`:  `auto`, `eager`, `sdpa`, `flash_attention_2`, or `sage`. (default: `auto`)
    *   `free_memory_after_generate`: Free VRAM after generation (default: `True`).
    *   `diffusion_steps`: Number of denoising steps (5-100, default: 20).
    *   `seed`: Random seed (default: 42).
    *   `cfg_scale`: Guidance scale (1.0-2.0, default: 1.3).
    *   `use_sampling`: Enable/disable sampling (default: `False`).
*   **Optional Parameters:**
    *   `voice_to_clone`: Audio input for voice cloning.
    *   `temperature`: Sampling temperature (0.1-2.0, default: 0.95).
    *   `top_p`: Nucleus sampling parameter (0.1-1.0, default: 0.95).
    *   `max_words_per_chunk`:  Maximum words per chunk (100-500, default: 250).

### 3. VibeVoice Multiple Speakers

*   Generates multi-speaker conversations.
*   **Speaker Format:**  Use `[N]:` notation (N = 1-4).
*   **Parameters:**
    *   `text`: Input text with speaker labels.
    *   `model`: VibeVoice-1.5B, VibeVoice-Large, or VibeVoice-Large-Quant-4Bit.
    *   `attention_type`: `auto`, `eager`, `sdpa`, `flash_attention_2` or `sage`. (default: `auto`)
    *   `free_memory_after_generate`: Free VRAM after generation (default: `True`).
    *   `diffusion_steps`: Number of denoising steps (5-100, default: 20).
    *   `seed`: Random seed (default: 42).
    *   `cfg_scale`: Guidance scale (1.0-2.0, default: 1.3).
    *   `use_sampling`: Enable/disable sampling (default: `False`).
*   **Optional Parameters:**
    *   `speaker1_voice` to `speaker4_voice`: Audio inputs for voice cloning.
    *   `temperature`: Sampling temperature (0.1-2.0, default: 0.95).
    *   `top_p`: Nucleus sampling parameter (0.1-1.0, default: 0.95).

### 4. VibeVoice Free Memory

*   Manually frees VibeVoice models from memory.
*   **Input:** `audio` (connect audio output).
*   **Output:** `audio` (passes input unchanged).
*   **Use Case:** Control VRAM/RAM usage.

## Multi-Speaker Text Format

Use the `[N]:` notation for multi-speaker generation:

```
[1]: Hello, how are you today?
[2]: I'm doing great, thanks for asking!
[1]: That's wonderful to hear.
[3]: Hey everyone, mind if I join the conversation?
[2]: Not at all, welcome!
```

*   Maximum 4 speakers supported.
*   The system detects the number of speakers from your text.
*   Each speaker can have an optional voice sample for cloning.

## Model Information

### VibeVoice-1.5B

*   **Size:** ~5GB.
*   **Speed:** Faster.
*   **Quality:** Good for single-speaker tasks.

### VibeVoice-Large

*   **Size:** ~17GB.
*   **Speed:** Slower, but optimized.
*   **Quality:** Best quality, ideal for multi-speaker.
*   **Note:**  Latest official release from Microsoft.

### VibeVoice-Large-Quant-4Bit

*   **Size:** ~7GB.
*   **Speed:** Balanced inference.
*   **Quality:** Good quality with reduced VRAM usage.
*   **Note:** Quantized by DevParker.

Models are automatically downloaded to `ComfyUI/models/vibevoice/`.

## Generation Modes

### Deterministic Mode (Default)

*   `use_sampling = False`
*   Consistent and stable output.
*   Recommended for production.

### Sampling Mode

*   `use_sampling = True`
*   More variation in output, controlled by `temperature` and `top_p`.
*   Good for creative exploration.

## Voice Cloning

1.  Connect an audio node to the `voice_to_clone` input (single speaker).
2.  Connect to `speaker1_voice`, `speaker2_voice`, etc. (multi-speaker).
3.  The model attempts to match voice characteristics.

**Requirements for voice samples:**

*   Clear audio, minimal noise.
*   Minimum 3-10 seconds (30+ seconds recommended).
*   Automatically resampled to 24kHz.

## Pause Tags Support

### Overview

The VibeVoice wrapper includes a custom pause tag feature that allows you to insert silences between text segments. **This is NOT a standard Microsoft VibeVoice feature** - it's an original implementation of our wrapper to provide more control over speech pacing.

**Available from version 1.3.0**

### Usage

You can use two types of pause tags in your text:

*   `[pause]` - Inserts a 1-second silence (default)
*   `[pause:ms]` - Inserts a custom duration silence in milliseconds (e.g., `[pause:2000]` for 2 seconds)

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
- Text before a pause and text after a pause are processed separately
- The model cannot see across pause boundaries when generating speech
- This may affect prosody and intonation consistency
- Use pauses sparingly for best results

### How It Works

1.  The wrapper parses your text to find pause tags
2.  Text segments between pauses are processed independently
3.  Silence audio is generated for each pause duration
4.  All audio segments (speech and silence) are concatenated

### Best Practices

-   Use pauses at natural breaking points (end of sentences, paragraphs)
-   Avoid pauses in the middle of phrases where context is important
-   Test different pause durations to find what sounds most natural

## Tips for Best Results

1.  **Text Preparation:** Use proper punctuation and break up long texts.
2.  **Model Selection:** Choose models based on your needs.
3.  **Seed Management:** Save and reuse seeds for consistent character voices.
4.  **Performance:** Use GPU, and download models first.

## System Requirements

### Hardware

*   **Minimum:** 8GB VRAM (VibeVoice-1.5B)
*   **Recommended:** 17GB+ VRAM (VibeVoice-Large)
*   **RAM:** 16GB+ system memory.

### Software

*   Python 3.8+
*   PyTorch 2.0+
*   CUDA 11.8+ (for GPU)
*   Transformers 4.51.3+
*   ComfyUI (latest version).

## Troubleshooting

*   Ensure you are using ComfyUI's Python environment.
*   Try manual installation if automatic fails.
*   Restart ComfyUI after installation.
*   Check the troubleshooting section.

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

| Model                     | VRAM Usage | Context Length | Max Audio Duration |
| ------------------------- | ---------- | -------------- | ------------------ |
| VibeVoice-1.5B            | ~8GB      | 64K tokens      | ~90 minutes       |
| VibeVoice-Large           | ~17GB     | 32K tokens      | ~45 minutes       |
| VibeVoice-Large-Quant-4Bit | ~7GB      | 32K tokens      | ~45 minutes       |

## Known Limitations

*   Maximum 4 speakers in multi-speaker mode.
*   Best performance with English and Chinese text.
*   Some seeds may produce unstable output.
*   Limited control over background music generation.

## License

MIT License.  See the `LICENSE` file.
**Note**: The VibeVoice model itself is subject to Microsoft's licensing terms.

## Links

*   [Original VibeVoice Repository](https://github.com/microsoft/VibeVoice) - Official Microsoft VibeVoice repository (currently unavailable)

## Credits

*   **VibeVoice Model:** Microsoft Research.
*   **ComfyUI Integration:** Fabio Sarracino.
*   **Base Model:** Qwen2.5 architecture.

## Support

*   Check troubleshooting.
*   Review ComfyUI logs.
*   Ensure proper installation.
*   Open an issue with details.

## Contributing

*   Test thoroughly.
*   Follow the code style.
*   Update documentation.
*   Submit pull requests.

## Changelog

*   *(Refer to the original README for a full list.)*