# VibeVoice ComfyUI Nodes: Transform Text into Natural-Sounding Speech with Ease

Unleash the power of Microsoft's VibeVoice directly within ComfyUI, enabling high-quality text-to-speech generation, voice cloning, and dynamic multi-speaker conversations, all in one versatile package. Get the VibeVoice ComfyUI integration at the [original repo](https://github.com/Enemyx-net/VibeVoice-ComfyUI)!

## Key Features

*   🎤 **Single & Multi-Speaker TTS:** Generate realistic speech with up to 4 distinct voices.
*   🗣️ **Voice Cloning:** Easily clone voices from audio samples for personalized speech.
*   📚 **Flexible Text Input:** Load scripts from files or directly input text.
*   ⚙️ **Customizable Parameters:** Control temperature, sampling, and guidance scale for unique results.
*   💾 **Optimized Performance:** Benefit from adjustable diffusion steps, memory management tools, and Apple Silicon support for speed and efficiency.
*   🔌 **Node Chaining:** Easily connect nodes together for complex workflows.
*   ✨ **Pause Tag Support:** Enhance your speech pacing with custom `[pause]` tags.

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

3.  Restart ComfyUI. Dependencies will install automatically on first use.

## Core Nodes Overview

### 1. VibeVoice Load Text From File

*   **Function:** Loads text from `.txt` files.
*   **Output:** Text string ready for TTS nodes.

### 2. VibeVoice Single Speaker

*   **Function:** Generates speech from single-speaker text.
*   **Inputs:** Text, Model Selection, Optional Voice Cloning
*   **Parameters:** Control steps, temperature, seed, guidance and memory usage.

### 3. VibeVoice Multiple Speakers

*   **Function:** Creates multi-speaker conversations.
*   **Format:** Use `[N]:` notation for each speaker (up to 4).
*   **Inputs:** Text with speaker labels, optional voice samples, model selection.
*   **Parameters:** Similar to Single Speaker, tailored for conversations.

### 4. VibeVoice Free Memory

*   **Function:** Manually releases VibeVoice models from memory.
*   **Usage:** Connect to audio output to trigger memory cleanup.

## Multi-Speaker Text Format

Format your text for multi-speaker generation using this notation:

```
[1]: Hello, how are you today?
[2]: I'm doing great, thanks for asking!
```

## Model Information

*   **VibeVoice-1.5B:** Fast, great for single speakers.
*   **VibeVoice-Large:** Top quality, best for multi-speaker.
*   **VibeVoice-Large-Quant-4Bit:** Good quality, lower VRAM usage.

## Generation Modes

*   **Deterministic:** (Default) Consistent output (`use_sampling = False`).
*   **Sampling:** Variable output with `temperature` and `top_p` (`use_sampling = True`).

## Voice Cloning

1.  Connect an audio node to `voice_to_clone` (single speaker) or `speaker1_voice`, `speaker2_voice`, etc. (multi-speaker).
2.  Provide clear audio (min. 3-10 seconds, recommended at least 30 seconds) with minimal noise.

## Pause Tags Support

*   Use `[pause]` (1-second silence) or `[pause:ms]` (custom duration in milliseconds).
*   **Important:** Pauses may affect context understanding; use them judiciously.

## Tips for Best Results

1.  Prepare text with punctuation and breaks.
2.  Choose the right model for your needs.
3.  Manage seeds for consistent voices.
4.  Optimize performance (GPU recommended).

## System Requirements

*   **Minimum:** 8GB VRAM.
*   **Recommended:** 16GB+ system RAM, 17GB+ VRAM.
*   **Software:** Python 3.8+, PyTorch 2.0+, CUDA 11.8+ (for GPU), Transformers 4.51.3+, ComfyUI.

## Troubleshooting

*   Check ComfyUI logs and follow the troubleshooting steps from the original repo.

## Examples

```
# Single Speaker
Text: "Welcome to our presentation."
Model: VibeVoice-1.5B
```

```
# Two Speakers
[1]: Have you seen the new AI developments?
[2]: Yes, they're quite impressive!
```

## Performance Benchmarks

| Model                  | VRAM Usage | Context Length | Max Audio Duration |
|------------------------|------------|----------------|-------------------|
| VibeVoice-1.5B         | ~8GB | 64K tokens | ~90 minutes |
| VibeVoice-Large | ~17GB | 32K tokens | ~45 minutes |
| VibeVoice-Large-Quant-4Bit | ~7GB | 32K tokens | ~45 minutes |

## Known Limitations

*   Maximum 4 speakers.
*   Best with English and Chinese text.
*   Some seeds may produce unstable output.
*   Limited background music control.

## License

MIT License. See the `LICENSE` file. VibeVoice model is subject to Microsoft's licensing terms.

## Credits

*   **VibeVoice Model:** Microsoft Research.
*   **ComfyUI Integration:** Fabio Sarracino.
*   **Base Model:** Qwen2.5 architecture.

## Support

*   Refer to the troubleshooting section.
*   Check ComfyUI logs.
*   Ensure proper installation.
*   Open an issue on the original repo with detailed info.

## Contributing

Please test changes, follow code style, update documentation, and submit pull requests.

## Changelog

A detailed changelog is included in the original README.