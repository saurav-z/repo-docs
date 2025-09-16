# VibeVoice ComfyUI Nodes: Unleash High-Quality AI Voice Synthesis in Your Workflows

**Elevate your ComfyUI creations with the power of Microsoft's VibeVoice, generating stunningly realistic single and multi-speaker speech directly within your workflows.**

[Go to the original repository](https://github.com/Enemyx-net/VibeVoice-ComfyUI)

## Key Features

*   🎤 **Single Speaker TTS:** Create natural-sounding speech with optional voice cloning.
*   👥 **Multi-Speaker Conversations:** Generate dynamic dialogues with up to 4 unique voices.
*   🔈 **Voice Cloning:** Personalize your voices by cloning from audio samples.
*   📝 **Text Input Flexibility:** Load scripts from text files or directly input text.
*   ✂️ **Automatic Text Chunking:** Seamlessly handles long texts with configurable chunk sizes.
*   ⏸️ **Custom Pause Tags:** Insert silences using `[pause]` and `[pause:ms]` for precise pacing.
*   🔄 **Node Chaining:** Connect multiple VibeVoice nodes for complex audio workflows.
*   🛑 **Interruption Support:** Cancel operations before or between generations.
*   🚀 **Model Variants:** Choose from VibeVoice-1.5B, VibeVoice-Large, and VibeVoice-Large-Quant-4Bit.
*   ⚙️ **Performance Controls:** Configure temperature, sampling, guidance scale, and diffusion steps.
*   ⚡ **Optimization:** Utilize attention mechanisms (auto, eager, sdpa, flash\_attention\_2, sage), VRAM management, and 4-bit quantization.
*   🍎 **Apple Silicon Support:** Benefit from native GPU acceleration via MPS on M1/M2/M3 Macs.
*   📦 **Self-Contained:** No external dependencies; all code is embedded.
*   💻 **Cross-Platform:** Compatible with Windows, Linux, and macOS.

## Video Demo

[Watch the demo video](https://www.youtube.com/watch?v=fIBMepIBKhI)

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
3.  Restart ComfyUI; the nodes will automatically install requirements on their first use.

## Available Nodes

### 1. VibeVoice Load Text From File

*   **Function:** Loads text content from `.txt` files.
*   **Input:** Text file.
*   **Output:** Text string for TTS nodes.

### 2. VibeVoice Single Speaker

*   **Function:** Generates speech from text using a single voice.
*   **Inputs:**
    *   `text`: Text to convert to speech (from Load Text or direct input).
    *   `voice_to_clone`: (Optional) Audio input for voice cloning.
*   **Parameters:**
    *   `model`: VibeVoice-1.5B, VibeVoice-Large, or VibeVoice-Large-Quant-4Bit.
    *   `attention_type`: (default: auto).
    *   `free_memory_after_generate`: (default: True)
    *   `diffusion_steps`: (5-100, default: 20).
    *   `seed`: (default: 42).
    *   `cfg_scale`: (1.0-2.0, default: 1.3).
    *   `use_sampling`: (default: False).
    *   `temperature`: (0.1-2.0, default: 0.95)
    *   `top_p`: (0.1-1.0, default: 0.95)
    *   `max_words_per_chunk`: (100-500, default: 250)
*   **Output:** Audio.

### 3. VibeVoice Multiple Speakers

*   **Function:** Generates multi-speaker conversations with distinct voices.
*   **Speaker Format:** Use `[N]:` notation, where N is 1-4.
    *   Example: `[1]: Hello, [2]: Hi there!`
*   **Inputs:**
    *   `text`: Text with speaker labels.
    *   `speaker1_voice` to `speaker4_voice`: (Optional) Audio inputs for voice cloning.
*   **Parameters:**
    *   `model`: VibeVoice-1.5B, VibeVoice-Large, or VibeVoice-Large-Quant-4Bit.
    *   `attention_type`: (default: auto).
    *   `free_memory_after_generate`: (default: True)
    *   `diffusion_steps`: (5-100, default: 20).
    *   `seed`: (default: 42).
    *   `cfg_scale`: (1.0-2.0, default: 1.3).
    *   `use_sampling`: (default: False).
    *   `temperature`: (0.1-2.0, default: 0.95)
    *   `top_p`: (0.1-1.0, default: 0.95)
*   **Output:** Audio.

### 4. VibeVoice Free Memory

*   **Function:** Manually frees VibeVoice models from memory.
*   **Input:** `audio` - connect audio output to trigger cleanup.
*   **Output:** `audio` - passes through the input audio.

## Multi-Speaker Text Format

Format your text using `[N]:` notation for multi-speaker generation:

```
[1]: Hello, how are you today?
[2]: I'm doing great, thanks!
```

**Important Notes:**

*   Use `[1]:`, `[2]:`, `[3]:`, `[4]:` for speaker labels.
*   Supports up to 4 speakers.
*   The system automatically detects the number of speakers.
*   Each speaker can have an optional voice sample for cloning.

## Model Information

### VibeVoice-1.5B

*   **Size:** ~5GB download
*   **Speed:** Faster inference
*   **Quality:** Good for single speaker tasks
*   **Use Case:** Quick prototyping, single voices

### VibeVoice-Large

*   **Size:** ~17GB download
*   **Speed:** Optimized for better quality, but slower inference
*   **Quality:** Best available quality
*   **Use Case:** Highest quality production, multi-speaker conversations
*   **Note:** Latest official release from Microsoft.

### VibeVoice-Large-Quant-4Bit

*   **Size:** ~7GB download
*   **Speed:** Balanced inference
*   **Quality:** Good quality
*   **Use Case:** Good quality production with less VRAM, multi-speaker conversations
*   **Note:** Quantized by DevParker

Models are automatically downloaded on first use and cached in `ComfyUI/models/vibevoice/`.

## Generation Modes

### Deterministic Mode (Default)

*   `use_sampling = False`
*   Produces consistent, stable output.
*   Recommended for production use.

### Sampling Mode

*   `use_sampling = True`
*   More variation in output.
*   Uses `temperature` and `top_p` parameters.
*   Good for creative exploration.

## Voice Cloning

1.  Connect an audio node to `voice_to_clone` (Single Speaker) or `speakerN_voice` (Multi-Speaker).
2.  The model attempts to match voice characteristics.

**Voice Sample Requirements:**

*   Clear audio, minimal background noise.
*   Minimum 3–10 seconds; 30 seconds recommended for best results.
*   Automatically resampled to 24kHz.

## Pause Tags Support

### Overview

The VibeVoice wrapper includes a custom pause tag feature for speech pacing control.

### Usage

Use two types of pause tags:

*   `[pause]` - 1-second silence (default)
*   `[pause:ms]` - Custom duration (e.g., `[pause:2000]` for 2 seconds)

### Examples

```
Welcome. [pause] Today's topic is AI. [pause:500] Let's begin!
```

```
[1]: Hello [pause] how are you today?
[2]: I'm doing great! [pause:500]
```

### Important Notes

⚠️ **Context Limitation Warning:**  Text before and after pauses is processed separately.

### Best Practices

*   Use pauses at natural breaking points.
*   Avoid pauses in the middle of phrases.
*   Test different durations.

## Tips for Best Results

1.  **Text Preparation:** Use punctuation, break long texts into paragraphs, and use clear speaker transitions for multi-speaker.
2.  **Model Selection:** 1.5B for speed, Large for quality/multi-speaker, Large-Quant-4Bit for a balance of quality and memory usage.
3.  **Seed Management:** Save good seeds for consistent character voices.
4.  **Performance:** GPU recommended for faster inference.

## System Requirements

### Hardware

*   **Minimum:** 8GB VRAM (VibeVoice-1.5B)
*   **Recommended:** 17GB+ VRAM (VibeVoice-Large)
*   **RAM:** 16GB+ system memory

### Software

*   Python 3.8+
*   PyTorch 2.0+
*   CUDA 11.8+ (for GPU acceleration)
*   Transformers 4.51.3+
*   ComfyUI (latest version)

## Troubleshooting

*   **Installation:** Ensure ComfyUI's Python environment and restart ComfyUI.
*   **Generation:** Use deterministic mode for unstable voices, check text formatting, and ensure speaker numbers are sequential.
*   **Memory:** Consider model size, use the 1.5B model, or manually manage memory with the Free Memory node.

## Examples

### Single Speaker

```
Text: "Welcome. We'll explore AI."
Model: VibeVoice-1.5B
cfg_scale: 1.3
use_sampling: False
```

### Two Speakers

```
[1]: Have you seen AI?
[2]: Yes, quite impressive!
```

### Four Speaker Conversation

```
[1]: Welcome to our meeting.
[2]: Thanks!
[3]: Glad to be here.
[4]: Looking forward to it.
```

## Performance Benchmarks

| Model                  | VRAM Usage | Context Length | Max Audio Duration |
|------------------------|------------|----------------|-------------------|
| VibeVoice-1.5B         | ~8GB       | 64K tokens     | ~90 minutes        |
| VibeVoice-Large        | ~17GB      | 32K tokens     | ~45 minutes        |
| VibeVoice-Large-Quant-4Bit | ~7GB      | 32K tokens     | ~45 minutes        |

## Known Limitations

*   Maximum 4 speakers in multi-speaker mode.
*   Best with English and Chinese.
*   Some seeds may produce unstable output.
*   Limited background music control.

## License

This ComfyUI wrapper is released under the MIT License. See the LICENSE file.

**Note:** The VibeVoice model is subject to Microsoft's licensing terms: for research purposes only; consult Microsoft's VibeVoice repository for full model license details.

## Links

*   [Original VibeVoice Repository](https://github.com/microsoft/VibeVoice) - Official Microsoft VibeVoice repository (currently unavailable)

## Credits

*   **VibeVoice Model:** Microsoft Research
*   **ComfyUI Integration:** Fabio Sarracino
*   **Base Model:** Built on Qwen2.5 architecture

## Support

1.  Check troubleshooting.
2.  Review ComfyUI logs.
3.  Ensure VibeVoice is properly installed.
4.  Open an issue with error information.

## Contributing

1.  Test changes thoroughly.
2.  Follow code style.
3.  Update documentation.
4.  Submit pull requests.

## Changelog

*(Condensed and summarized for brevity)*

### Version 1.3.0

*   Added custom pause tags for speech pacing control.
### Version 1.2.5
*   Bug fixing
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
### Version 1.0.1

*   Fixed issue with line breaks in speaker text (both single and multi-speaker nodes)
*   Improved text formatting handling for all generation modes
### Version 1.0.0

*   Initial release