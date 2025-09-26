# VibeVoice ComfyUI: Unleash High-Quality Text-to-Speech in Your ComfyUI Workflows

Effortlessly generate natural, high-quality speech and multi-speaker conversations directly within ComfyUI using [VibeVoice ComfyUI](https://github.com/Enemyx-net/VibeVoice-ComfyUI), a powerful integration for Microsoft's VibeVoice text-to-speech model.

**Key Features:**

*   🗣️ **Single & Multi-Speaker TTS:** Create realistic speech with up to 4 distinct voices.
*   🎤 **Voice Cloning:** Clone voices from audio samples for personalized audio.
*   🎧 **LoRA Support:** Fine-tune voices with custom LoRA adapters for unique styles.
*   💾 **Memory Optimization:** Choose between manual & automatic memory management.
*   🍎 **Apple Silicon Support:** Native GPU acceleration on M1/M2/M3 Macs via MPS.
*   🧩 **Flexible Control:** Adjust parameters like temperature, sampling, and attention.
*   📝 **Text File Loading:** Load scripts directly from text files for easy workflow integration.
*   ⏹️ **Interruption Support:** Cancel generation at any time.
*   ⚡ **Multiple Models:** Use VibeVoice-1.5B (fastest, lowest memory), VibeVoice-Large (best quality, ~17GB VRAM), or VibeVoice-Large-Quant-4Bit (balanced, ~7GB VRAM)

## Key Benefits

*   **Seamless Integration:** Integrate VibeVoice directly into your existing ComfyUI workflows, saving time and effort.
*   **High-Quality Audio:** Generate realistic speech output that sounds great.
*   **Voice Customization:** Clone voices or customize voices with LoRA adapters for perfect results.
*   **Optimization for Speed and Memory:** Choose attention mechanisms and model variants to suit your hardware.
*   **Cross-Platform Compatibility:** Runs on Windows, Linux, and macOS.

## Installation

### Automatic Installation (Recommended)

1.  Navigate to your ComfyUI custom nodes folder: `cd ComfyUI/custom_nodes`
2.  Clone the repository: `git clone https://github.com/Enemyx-net/VibeVoice-ComfyUI`
3.  Restart ComfyUI - the nodes will automatically install dependencies on first use.

## Available Nodes

### 1.  VibeVoice Load Text From File
    *   Loads text content from files (.txt) in ComfyUI input/output/temp directories.
    *   **Output**: Text string for TTS nodes.

### 2.  VibeVoice Single Speaker
    *   Generates speech from text using a single voice.
    *   **Input**: Text or connection from Load Text node.
    *   **Models**: VibeVoice-1.5B, VibeVoice-Large, or VibeVoice-Large-Quant-4Bit.
    *   **Optional**: Voice cloning, LoRA, temperature, sampling parameters.

### 3.  VibeVoice Multiple Speakers
    *   Generates multi-speaker conversations with distinct voices.
    *   **Input**: Text with speaker labels (e.g., `[1]: Hello. [2]: Hi`).
    *   **Input**: Text with speaker labels (e.g., \[1]: Hello. \[2]: Hi).
    *   **Models**: VibeVoice-1.5B, VibeVoice-Large, or VibeVoice-Large-Quant-4Bit.
    *   **Optional**: Voice cloning (per speaker), LoRA, temperature, sampling parameters.

### 4.  VibeVoice Free Memory
    *   Manually frees all loaded VibeVoice models from memory.
    *   **Input**: `audio` - Connect audio output to trigger cleanup.
    *   **Output**: `audio` - Passes through the input audio unchanged.
    *   **Use Case**: Insert between nodes to free VRAM/RAM at specific workflow points.

### 5.  VibeVoice LoRA
    *   Configure and load custom LoRA adapters for fine-tuned VibeVoice models.
    *   **LoRA Location**: Place your LoRA folders in `ComfyUI/models/vibevoice/loras/`.
    *   **Parameters**: LoRA selection, strength, and component toggles.
    *   **Output**: LoRA configuration to connect to speaker nodes.

## Model Information

*   **VibeVoice-1.5B:** Fastest inference, good for single-speaker tasks, ~8GB VRAM.
*   **VibeVoice-Large:** Best quality, multi-speaker conversations, ~17GB VRAM.
*   **VibeVoice-Large-Quant-4Bit:** Balanced performance, good quality with lower VRAM usage, ~7GB VRAM.

*Models are automatically downloaded on first use and cached in `ComfyUI/models/vibevoice/`.*

## Voice Cloning

1.  Connect an audio node to the `voice_to_clone` input (single speaker) or `speakerX_voice` (multi-speaker).
2.  The model will attempt to match the voice characteristics.

    *   Requires clear audio with minimal noise (30+ seconds recommended).

## LoRA Support

*   **How to Use:** Use the "VibeVoice LoRA" node to select and configure custom LoRA adapters.
*   **LoRA Location:** Place adapter folders in `ComfyUI/models/vibevoice/loras/`.
*   **Training:** Fine-tune your own LoRA adapters using the [VibeVoice Fine-tuning repository](https://github.com/voicepowered-ai/VibeVoice-finetuning).

## Pause Tags

*   Insert silences with `[pause]` (1 second) or `[pause:ms]` (custom duration).
*   **Note:** Pause tags split the text into chunks which may impact context and intonation consistency.

## Troubleshooting

*   Ensure ComfyUI is in the correct Python environment.
*   Check ComfyUI logs for any error messages.
*   Double-check text formatting for multi-speaker conversations.
*   Use deterministic mode for stable output.

## Performance Benchmarks

| Model                    | VRAM Usage | Context Length | Max Audio Duration |
| -------------------------- | ---------- | -------------- | ------------------ |
| VibeVoice-1.5B            | ~8GB       | 64K tokens     | ~90 minutes        |
| VibeVoice-Large           | ~17GB      | 32K tokens     | ~45 minutes        |
| VibeVoice-Large-Quant-4Bit | ~7GB       | 32K tokens     | ~45 minutes        |

## Links

*   [Original VibeVoice Repository](https://github.com/Enemyx-net/VibeVoice-ComfyUI) - Official Microsoft VibeVoice repository (currently unavailable)

## License

This ComfyUI wrapper is released under the MIT License. See LICENSE file for details.

**Note:** The VibeVoice model itself is subject to Microsoft's licensing terms:
*   VibeVoice is for research purposes only
*   Check Microsoft's VibeVoice repository for full model license details