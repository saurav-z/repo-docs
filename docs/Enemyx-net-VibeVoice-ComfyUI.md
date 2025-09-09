# VibeVoice ComfyUI Nodes: High-Fidelity Text-to-Speech within ComfyUI

Create stunning, natural-sounding speech and multi-speaker conversations directly within your ComfyUI workflows using Microsoft's VibeVoice text-to-speech model. [See the original repository](https://github.com/Enemyx-net/VibeVoice-ComfyUI).

## Key Features

*   🎤 **Single & Multi-Speaker TTS:** Generate voices or conversations with up to four distinct speakers.
*   🔊 **Voice Cloning:** Clone voices from audio samples for personalized speech.
*   📝 **Text Input Flexibility:** Load scripts from text files or directly input text.
*   ⚙️ **Model Options:** Choose from three VibeVoice models for performance and quality, including a 4-bit quantized model for reduced VRAM usage.
*   ⚡ **Performance Optimization:** Optimize generation with attention mechanisms, adjustable diffusion steps, and memory management tools.
*   🍎 **Apple Silicon Support:** Harness native GPU acceleration on M1/M2/M3 Macs via MPS.
*   🔄 **Workflow Integration:** Chain nodes and utilize a dedicated memory-freeing node for complex workflows.

## Getting Started

### Installation (Automatic - Recommended)

1.  Navigate to your ComfyUI custom nodes directory:

    ```bash
    cd ComfyUI/custom_nodes
    ```
2.  Clone the repository:

    ```bash
    git clone https://github.com/Enemyx-net/VibeVoice-ComfyUI
    ```
3.  Restart ComfyUI; dependencies will install automatically on first use.

## Available Nodes

### 1. VibeVoice Load Text From File

*   **Functionality**: Loads text from `.txt` files within ComfyUI's input/output/temp directories.
*   **Output**: A text string for use in TTS nodes.

### 2. VibeVoice Single Speaker

*   **Functionality**: Generates speech from input text using a single voice.
*   **Inputs**: Text (direct input or from "Load Text" node), optional voice cloning audio.
*   **Model Selection**: VibeVoice-1.5B, VibeVoice-Large, or VibeVoice-Large-Quant-4Bit.
*   **Parameters**: `text`, `model`, `attention_type`, `free_memory_after_generate`, `diffusion_steps`, `seed`, `cfg_scale`, `use_sampling`, `temperature`, `top_p`.
*   **Optional**: `voice_to_clone`.

### 3. VibeVoice Multiple Speakers

*   **Functionality**: Generates multi-speaker conversations, supporting up to four speakers.
*   **Format**: Use `[N]:` (e.g., `[1]: Hello...`) to specify speakers.
*   **Inputs**: Text (with speaker labels), optional voice cloning audio for each speaker.
*   **Model Selection**: VibeVoice-1.5B, VibeVoice-Large, or VibeVoice-Large-Quant-4Bit.
*   **Parameters**: `text`, `model`, `attention_type`, `free_memory_after_generate`, `diffusion_steps`, `seed`, `cfg_scale`, `use_sampling`, `temperature`, `top_p`.
*   **Optional**: `speaker1_voice` to `speaker4_voice`.

### 4. VibeVoice Free Memory

*   **Functionality**: Manually releases VibeVoice model memory.
*   **Input**:  `audio` - Trigger cleanup by connecting audio output.
*   **Output**: `audio` - Passes the audio through unchanged.
*   **Usage**:  Insert between nodes to free VRAM/RAM at specific points in your workflow.

## Text Formatting for Multi-Speaker

Use the `[N]:` format:

```
[1]: Hello, how are you?
[2]: I'm fine, thanks!
```

*   Labels: `[1]:`, `[2]:`, `[3]:`, `[4]:`
*   Up to 4 speakers supported
*   Speaker voice cloning is optional.

## Model Information

### VibeVoice-1.5B

*   Size: ~5GB
*   Speed: Faster
*   Quality: Good for single speaker
*   Use Case: Quick prototyping, single voices

### VibeVoice-Large

*   Size: ~17GB
*   Speed: Slower inference, optimized
*   Quality: Best available
*   Use Case: Highest quality, multi-speaker conversations

### VibeVoice-Large-Quant-4Bit

*   Size: ~7GB
*   Speed: Balanced
*   Quality: Good
*   Use Case: Good quality with less VRAM, multi-speaker conversations

Models download automatically to `ComfyUI/models/vibevoice/` on first use.

## Generation Modes

### Deterministic Mode

*   `use_sampling = False`
*   Consistent output.
*   Recommended for production.

### Sampling Mode

*   `use_sampling = True`
*   More varied output.
*   Adjust `temperature` and `top_p` for control.
*   Best for creative exploration.

## Voice Cloning Guide

1.  Connect an audio node to:
    *   `voice_to_clone` (single speaker)
    *   `speaker1_voice`, `speaker2_voice`, etc. (multi-speaker)
2.  Ensure clear audio with minimal noise, at least 3-10 seconds (30 seconds recommended).
3.  Audio is automatically resampled to 24kHz.

## Tips for Best Results

1.  **Text Preparation**: Use proper punctuation; break up long text.
2.  **Model Selection**: Choose model based on needs, VRAM availability.
3.  **Seed Management**: Save good seeds for character consistency.
4.  **Performance**: GPU recommended; the initial run downloads models.

## System Requirements

### Hardware

*   **Minimum**: 8GB VRAM for VibeVoice-1.5B
*   **Recommended**: 17GB+ VRAM for VibeVoice-Large
*   **RAM**: 16GB+ system memory

### Software

*   Python 3.8+
*   PyTorch 2.0+
*   CUDA 11.8+ (for GPU acceleration)
*   Transformers 4.51.3+
*   ComfyUI (latest version)

## Troubleshooting

*   **Installation**: Use ComfyUI's environment; try manual installation.
*   **Generation**: Deterministic mode may stabilize voices.
*   **Multi-Speaker**: Check formatting.
*   **Memory**: Use the right model for available VRAM; the Free Memory Node helps.

## Examples

### Single Speaker

```
Text: "Welcome..."
Model: VibeVoice-1.5B
cfg_scale: 1.3
use_sampling: False
```

### Two Speakers

```
[1]: "Have you..."
[2]: "Yes..."
```

### Four Speaker Conversation

```
[1]: "Welcome..."
[2]: "Thanks..."
[3]: "Glad..."
[4]: "Looking..."
```

## Performance Benchmarks

| Model                      | VRAM Usage | Context Length | Max Audio Duration |
| -------------------------- | ---------- | -------------- | ------------------ |
| VibeVoice-1.5B             | ~8GB      | 64K tokens     | ~90 minutes         |
| VibeVoice-Large            | ~17GB     | 32K tokens     | ~45 minutes         |
| VibeVoice-Large-Quant-4Bit | ~7GB      | 32K tokens     | ~45 minutes         |

## Known Limitations

*   Maximum 4 speakers
*   Best for English/Chinese
*   Some seeds may be unstable.
*   Limited background music control.

## License

MIT License. See the `LICENSE` file.
**Note:** VibeVoice model itself is for research purposes only, subject to Microsoft's licensing terms.

## Links

*   [Original VibeVoice Repository](https://github.com/microsoft/VibeVoice) (currently unavailable)

## Credits

*   **VibeVoice Model**: Microsoft Research
*   **ComfyUI Integration**: Fabio Sarracino
*   **Base Model**: Qwen2.5

## Support

1.  Check the Troubleshooting section.
2.  Review ComfyUI logs.
3.  Ensure VibeVoice is installed.
4.  Open an issue with detailed error information.

## Contributing

Contributions are welcome! Follow existing code style and submit pull requests with clear descriptions.

## Changelog

*   **1.2.2**: Added 4-bit quantized model support
*   **1.2.1**: Bug Fixes
*   **1.2.0**: MPS Support for Apple Silicon
*   **1.1.1**: Universal Transformers Compatibility
*   **1.1.0**: Updated download URL
*   **1.0.9**: Embedded VibeVoice code
*   **1.0.8**: BFloat16 Compatibility Fix
*   **1.0.7**: Added interruption handler; bug fixes
*   **1.0.6**: Fixed direct audio input from other nodes
*   **1.0.5**: Added official VibeVoice-Large model
*   **1.0.4**: Improved tokenizer dependency
*   **1.0.3**: Added `attention_type` and `diffusion_steps` parameters
*   **1.0.2**: Added `free_memory_after_generate` and Free Memory Node
*   **1.0.1**: Fixed line breaks in speaker text
*   **1.0.0**: Initial Release