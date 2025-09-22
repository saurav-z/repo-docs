# VibeVoice ComfyUI: Transform Text into Natural-Sounding Speech

**Elevate your ComfyUI workflows with the power of VibeVoice, effortlessly generating high-quality, realistic speech with single and multi-speaker support directly within your projects.**

[Click here to visit the original VibeVoice-ComfyUI repository](https://github.com/Enemyx-net/VibeVoice-ComfyUI)

## Key Features

*   🎤 **Single & Multi-Speaker TTS:** Generate natural speech for up to 4 speakers.
*   🗣️ **Voice Cloning:** Clone voices from audio samples for personalized audio.
*   📝 **Flexible Input:** Load text from files or input directly.
*   🧠 **Smart Chunking:** Handles long texts with automatic chunking.
*   ⏱️ **Custom Pauses:** Insert silences using custom pause tags.
*   🔄 **Workflow Integration:** Node chaining for complex audio generation.
*   🚀 **Model Options:** Choose between VibeVoice 1.5B, VibeVoice-Large, and VibeVoice-Large-Quant-4Bit for speed and quality trade-offs.
*   ⚙️ **Optimized Performance:** Choose between various attention mechanisms, adjust diffusion steps, and manage VRAM.
*   🍎 **Apple Silicon Support:** Native GPU acceleration via MPS on M1/M2/M3 Macs.
*   💾 **Memory Management:** Control VRAM usage with the Free Memory Node.

## Installation

### Automatic Installation (Recommended)

1.  Clone the repository into your ComfyUI custom nodes folder:

    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/Enemyx-net/VibeVoice-ComfyUI
    ```

2.  Restart ComfyUI - the nodes will automatically install requirements on first use.

## Available Nodes

### 1. VibeVoice Load Text From File

*   **Description:** Loads text from .txt files.
*   **Input:** File path.
*   **Output:** Text string.

### 2. VibeVoice Single Speaker

*   **Description:** Generates speech from a single voice.
*   **Input:** Text, Model Selection, and optional parameters (voice cloning, sampling, etc.).
*   **Output:** Audio.
    *   **Parameters** (in order):
        *   `text`: Input text
        *   `model`: VibeVoice-1.5B, VibeVoice-Large or VibeVoice-Large-Quant-4Bit
        *   `attention_type`: auto, eager, sdpa, flash_attention_2 or sage (default: auto)
        *   `free_memory_after_generate`: Free VRAM after generation (default: True)
        *   `diffusion_steps`: Number of denoising steps (5-100, default: 20)
        *   `seed`: Random seed (default: 42)
        *   `cfg_scale`: Classifier-free guidance (1.0-2.0, default: 1.3)
        *   `use_sampling`: Enable/disable deterministic generation (default: False)
    *   **Optional Parameters:**
        *   `voice_to_clone`: Audio input for voice cloning
        *   `temperature`: Sampling temperature (0.1-2.0, default: 0.95)
        *   `top_p`: Nucleus sampling parameter (0.1-1.0, default: 0.95)
        *   `max_words_per_chunk`: Maximum words per chunk for long texts (100-500, default: 250)

### 3. VibeVoice Multiple Speakers

*   **Description:** Generates multi-speaker conversations.
*   **Input:** Text with speaker labels (`[N]:`), Model Selection, and optional parameters.
*   **Output:** Audio.
    *   **Parameters** (in order):
        *   `text`: Input text with speaker labels
        *   `model`: VibeVoice-1.5B, VibeVoice-Large or VibeVoice-Large-Quant-4Bit
        *   `attention_type`: auto, eager, sdpa, flash_attention_2 or sage (default: auto)
        *   `free_memory_after_generate`: Free VRAM after generation (default: True)
        *   `diffusion_steps`: Number of denoising steps (5-100, default: 20)
        *   `seed`: Random seed (default: 42)
        *   `cfg_scale`: Classifier-free guidance (1.0-2.0, default: 1.3)
        *   `use_sampling`: Enable/disable deterministic generation (default: False)
    *   **Optional Parameters:**
        *   `speaker1_voice` to `speaker4_voice`: Audio inputs for voice cloning
        *   `temperature`: Sampling temperature (0.1-2.0, default: 0.95)
        *   `top_p`: Nucleus sampling parameter (0.1-1.0, default: 0.95)

### 4. VibeVoice Free Memory

*   **Description:** Manually frees VibeVoice models from memory.
*   **Input:** Audio (trigger).
*   **Output:** Audio (passthrough).

## Multi-Speaker Text Format

Use `[N]:` labels for speakers (up to 4):

```
[1]: Hello, how are you?
[2]: I'm great, thanks!
```

*   Maximum 4 speakers supported.
*   System automatically detects speakers.
*   Optional voice cloning for each speaker.

## Model Information

### VibeVoice-1.5B

*   **Size:** ~5GB.
*   **Speed:** Faster.
*   **Quality:** Good for single-speaker.
*   **Use Case:** Prototyping, single voices.

### VibeVoice-Large

*   **Size:** ~17GB.
*   **Speed:** Slower.
*   **Quality:** Best quality.
*   **Use Case:** Highest quality, multi-speaker.

### VibeVoice-Large-Quant-4Bit

*   **Size:** ~7GB
*   **Speed:** Balanced
*   **Quality:** Good quality.
*   **Use Case:** Good quality production with less VRAM, multi-speaker conversations
*   **Note:** Quantized by DevParker

Models download to `ComfyUI/models/vibevoice/`.

## Generation Modes

### Deterministic Mode (Default)

*   `use_sampling = False`
*   Consistent output.
*   Recommended for production.

### Sampling Mode

*   `use_sampling = True`
*   More varied output.
*   Use temperature and top\_p.
*   Exploration.

## Voice Cloning

1.  Connect an audio node to `voice_to_clone` (single speaker) or `speakerN_voice` (multi-speaker).
2.  Clear audio with minimal noise is best, with a sample that is at least 30 seconds in length.
3.  Automatically resampled to 24kHz

## Pause Tags Support

*   Custom pause tags for pacing control.
    *   `[pause]` - 1-second silence.
    *   `[pause:ms]` - Custom duration in milliseconds (e.g., `[pause:2000]`).
*   Wrapper feature, not a standard VibeVoice feature.

**Context Limitation Warning**: The model's context is limited by the chunk before the pause.

## Tips for Best Results

1.  **Text Prep:** Use punctuation, break up long texts, and format multi-speaker text clearly.
2.  **Model Choice:** 1.5B for speed, Large for quality, Large-Quant-4Bit for low VRAM.
3.  **Seed:** Save good seeds.
4.  **Performance:** GPU is recommended.

## System Requirements

*   **Minimum:** 8GB VRAM (1.5B), 16GB System Memory.
*   **Recommended:** 17GB+ VRAM (Large), 16GB+ System Memory.
*   **Software:** Python 3.8+, PyTorch 2.0+, CUDA 11.8+ (GPU), Transformers 4.51.3+, ComfyUI.

## Troubleshooting

*   Installation: Ensure ComfyUI's environment, restart ComfyUI.
*   Generation: Deterministic mode, correct multi-speaker format.
*   Memory: Consider model size, use Free Memory Node.

## Examples

```
Single Speaker:
Text: "Welcome. Today we'll explore AI."
Model: VibeVoice-1.5B
cfg_scale: 1.3
use_sampling: False
```

```
Two Speakers:
[1]: New AI?
[2]: Impressive!
```

## Performance Benchmarks

| Model                     | VRAM Usage | Context Length | Max Audio Duration |
| :------------------------ | :---------- | :------------- | :----------------- |
| VibeVoice-1.5B          | ~8GB        | 64K tokens     | ~90 minutes        |
| VibeVoice-Large           | ~17GB       | 32K tokens     | ~45 minutes        |
| VibeVoice-Large-Quant-4Bit | ~7GB        | 32K tokens     | ~45 minutes        |

## Known Limitations

*   Max 4 speakers.
*   Best with English/Chinese.
*   Some seeds unstable.
*   No background music control.

## License

MIT License. See the LICENSE file. VibeVoice model is subject to Microsoft's licensing terms.

## Credits

*   **VibeVoice Model:** Microsoft Research
*   **ComfyUI Integration:** Fabio Sarracino

## Support

Check troubleshooting, ComfyUI logs, ensure installation, or open an issue.

## Contributing

Test, follow code style, update documentation, and submit pull requests.

## Changelog

(See original README for detailed changelog)