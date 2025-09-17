# VibeVoice ComfyUI Nodes: Turn Text into Natural-Sounding Speech 🎙️

This powerful ComfyUI integration brings Microsoft's cutting-edge VibeVoice text-to-speech technology directly to your workflows, enabling stunning voice synthesis with ease. [Check out the original repo!](https://github.com/Enemyx-net/VibeVoice-ComfyUI)

## Key Features

*   **Single and Multi-Speaker Support:** Generate voices from text with up to four speakers.
*   **Voice Cloning:** Replicate voices using audio samples for realistic output.
*   **Advanced Text Handling:** Load scripts from text files, and use custom pause tags to control pacing.
*   **Flexible Control:** Configure model choice, attention mechanisms, diffusion steps, temperature, sampling, and guidance scale.
*   **Optimized Performance:** Choose attention mechanisms, diffusion steps, and memory management options for optimal speed and VRAM usage.
*   **Broad Compatibility:** Works on Windows, Linux, and macOS (including Apple Silicon with MPS).
*   **Multi-Model Support:** Access to the fast 1.5B model, high-quality Large, and the memory-efficient Quant-4Bit model variants.

## Demo

[Watch the demo video](https://www.youtube.com/watch?v=fIBMepIBKhI) to see VibeVoice in action!

## Installation

### Automatic Installation (Recommended)

1.  **Clone the repository:**
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/Enemyx-net/VibeVoice-ComfyUI
    ```
2.  **Restart ComfyUI:** The necessary dependencies will automatically install on first use.

## Available Nodes

### 1. VibeVoice Load Text From File

*   **Function:** Loads text from .txt files.
*   **Output:** Text string.

### 2. VibeVoice Single Speaker

*   **Function:** Generates speech using a single voice.
*   **Input:** Text or connection from "Load Text From File" node.
*   **Voice Cloning:** Optional audio input.
*   **Key Parameters:**
    *   `model`: Choose from `VibeVoice-1.5B`, `VibeVoice-Large`, or `VibeVoice-Large-Quant-4Bit`.
    *   `attention_type`: Control attention mechanism (`auto`, `eager`, `sdpa`, `flash_attention_2`, `sage`).
    *   `diffusion_steps`: Control the denoising steps (5-100, default: 20)
    *   `free_memory_after_generate`: (Default: True)

### 3. VibeVoice Multiple Speakers

*   **Function:** Creates multi-speaker conversations.
*   **Speaker Format:** Use `[N]:` notation, with N (1-4) representing the speaker.
*   **Voice Assignment:** Optional voice samples per speaker.
*   **Key Parameters:** Identical to Single Speaker but includes optional `speakerN_voice` inputs for cloning.

### 4. VibeVoice Free Memory

*   **Function:** Manually frees VRAM.
*   **Input:** Connect audio to trigger memory cleanup.
*   **Output:** Passes the input audio through.

## Text Formatting for Multiple Speakers

Use the following format for multi-speaker generation:

```
[1]: Hello, how are you today?
[2]: I'm doing great, thanks for asking!
```

## Model Information

*   **VibeVoice-1.5B:** Fast, good for single speakers (~8GB VRAM).
*   **VibeVoice-Large:** Best quality, multi-speaker capable (~17GB VRAM).
*   **VibeVoice-Large-Quant-4Bit:** Balances quality and memory usage (~7GB VRAM).

## Generation Modes

*   **Deterministic Mode:** (`use_sampling = False`) Consistent output (default).
*   **Sampling Mode:** (`use_sampling = True`) For varied, creative output (uses `temperature` and `top_p`).

## Voice Cloning

1.  Connect an audio node to the `voice_to_clone` input (single speaker) or the `speakerN_voice` inputs (multi-speaker).
2.  Provide clear, noise-free audio (minimum 3–10 seconds, ideally 30+ seconds).

## Pause Tags Support

*   The wrapper allows you to insert silences:
    *   `[pause]` (1-second silence)
    *   `[pause:ms]` (custom duration in milliseconds, e.g., `[pause:2000]`)

*   **Important Note:** Pauses can affect prosody and context.

## Tips for Best Results

1.  **Text Preparation:** Use proper punctuation and break long texts.
2.  **Model Choice:** Select the right model based on needs (1.5B for speed, Large for quality).
3.  **Seed Management:** Save and reuse seeds for consistent voices.
4.  **Performance:** GPUs are recommended for quicker performance.

## System Requirements

*   **Hardware:**
    *   8GB+ VRAM (VibeVoice-1.5B)
    *   17GB+ VRAM recommended (VibeVoice-Large)
    *   16GB+ system RAM.
*   **Software:**
    *   Python 3.8+
    *   PyTorch 2.0+
    *   CUDA 11.8+ (for GPU acceleration)
    *   Transformers 4.51.3+
    *   ComfyUI (latest version)

## Troubleshooting

*   Check the troubleshooting section of the original README.
*   Review ComfyUI logs.
*   Ensure proper installation.

## Examples

Refer to the original README for single-speaker, two-speaker, and four-speaker examples.

## Performance Benchmarks

| Model                       | VRAM Usage | Context Length | Max Audio Duration |
| --------------------------- | ---------- | -------------- | ------------------ |
| VibeVoice-1.5B              | ~8GB       | 64K tokens     | ~90 minutes        |
| VibeVoice-Large             | ~17GB      | 32K tokens     | ~45 minutes        |
| VibeVoice-Large-Quant-4Bit  | ~7GB       | 32K tokens     | ~45 minutes        |

## Known Limitations

*   Maximum 4 speakers.
*   Best results with English and Chinese text.
*   Some seeds may produce unstable output.
*   Limited control over background music generation.

## License

MIT License. See the LICENSE file.

**Note:** The VibeVoice model is subject to Microsoft's licensing terms.

## Credits

*   VibeVoice Model: Microsoft Research
*   ComfyUI Integration: Fabio Sarracino
*   Base Model: Built on Qwen2.5 architecture

## Support

*   Consult the troubleshooting section.
*   Review ComfyUI logs.
*   Ensure proper VibeVoice installation.
*   Open an issue with detailed information.

## Contributing

*   Test changes thoroughly.
*   Follow code style.
*   Update documentation.
*   Submit pull requests.

## Changelog

See the original README for the detailed changelog.