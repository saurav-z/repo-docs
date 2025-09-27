# VibeVoice ComfyUI Nodes: Unleash High-Quality Text-to-Speech in Your Workflows

Bring the power of Microsoft's VibeVoice text-to-speech model directly into ComfyUI with these user-friendly nodes, enabling natural-sounding voice synthesis and multi-speaker conversations.  Check out the original repo [here](https://github.com/Enemyx-net/VibeVoice-ComfyUI)!

## Key Features

*   🎤 **Single Speaker TTS:** Generate realistic speech with optional voice cloning.
*   🗣️ **Multi-Speaker Conversations:** Create dynamic dialogues with up to 4 distinct speakers.
*   🔊 **Voice Cloning:** Clone voices from audio samples for personalized output.
*   🎨 **LoRA Support:** Fine-tune voices with custom LoRA adapters for unique characterizations.
*   ⏱️ **Voice Speed Control**: Adjust speech rate for dynamic pacing.
*   📝 **Text File Loading:** Import scripts directly from text files.
*   🧩 **Automatic Text Chunking:** Handle long texts effortlessly with configurable chunk size.
*   ⏸️ **Custom Pause Tags:** Insert silences with `[pause]` and `[pause:ms]` tags.
*   🔗 **Node Chaining:** Create complex workflows by connecting multiple nodes.
*   🛑 **Interruption Support:** Cancel operations before or between generations.
*   ⚡ **Performance Optimization:** Choose attention mechanisms and adjust diffusion steps for optimal speed and quality.
*   💾 **Memory Management:**  Toggle VRAM cleanup and manual memory control for streamlined workflows.
*   🍎 **Apple Silicon Support:** Native GPU acceleration via MPS on M1/M2/M3 Macs.
*   💾 **4-Bit Quantization:** Reduces memory usage with minimal quality loss.
*   📦 **Self-Contained:** Embedded VibeVoice code, no external dependencies
*   🔄 **Universal Compatibility:** Adaptive support for transformers v4.51.3+
*   🖥️ **Cross-Platform:** Works on Windows, Linux, and macOS

## Video Demo
<p align="center">
  <a href="https://www.youtube.com/watch?v=fIBMepIBKhI">
    <img src="https://img.youtube.com/vi/fIBMepIBKhI/maxresdefault.jpg" alt="VibeVoice ComfyUI Wrapper Demo" />
  </a>
  <br>
  <strong>Click to watch the demo video</strong>
</p>

## Installation

### Automatic Installation (Recommended)

1.  Clone this repository into your ComfyUI custom nodes folder:

    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/Enemyx-net/VibeVoice-ComfyUI
    ```

2.  Restart ComfyUI - the nodes will automatically install requirements on first use.

## Available Nodes

### 1.  VibeVoice Load Text From File

*   Loads text content from `.txt` files within ComfyUI's input/output/temp directories.
*   **Output**: Text string for TTS nodes.

### 2.  VibeVoice Single Speaker

*   Generates speech from text using a single voice.
*   **Input**: Direct text or connection from Load Text node.
*   **Models**: VibeVoice-1.5B, VibeVoice-Large, or VibeVoice-Large-Quant-4Bit.
*   **Voice Cloning**: Optional audio input for voice cloning.
*   **Parameters**:
    *   `text`: Input text.
    *   `model`: Model selection.
    *   `attention_type`: Attention mechanism (auto, eager, sdpa, flash\_attention\_2, sage).
    *   `free_memory_after_generate`: Free VRAM after generation.
    *   `diffusion_steps`: Denoising steps (5-100).
    *   `seed`: Random seed.
    *   `cfg_scale`: Classifier-free guidance (1.0-2.0).
    *   `use_sampling`: Enable sampling mode (creative exploration).
    *   `voice_to_clone`: (Optional) Audio input for voice cloning.
    *   `lora`: (Optional) LoRA configuration from VibeVoice LoRA node.
    *   `temperature`: (Optional) Sampling temperature.
    *   `top_p`: (Optional) Nucleus sampling parameter.
    *   `max_words_per_chunk`: (Optional) Max words per chunk.
    *   `voice_speed_factor`: (Optional) Speech rate adjustment (0.8-1.2).

### 3.  VibeVoice Multiple Speakers

*   Generates multi-speaker conversations with distinct voices.
*   **Speaker Format**: Use `[N]:` notation (1-4).
*   **Voice Assignment**: Optional voice samples for each speaker.
*   **Recommended Model**: VibeVoice-Large.
*   **Parameters**:
    *   `text`: Input text with speaker labels.
    *   `model`: Model selection.
    *   `attention_type`: Attention mechanism (auto, eager, sdpa, flash\_attention\_2, sage).
    *   `free_memory_after_generate`: Free VRAM after generation.
    *   `diffusion_steps`: Denoising steps (5-100).
    *   `seed`: Random seed.
    *   `cfg_scale`: Classifier-free guidance (1.0-2.0).
    *   `use_sampling`: Enable sampling mode.
    *   `speaker1_voice` to `speaker4_voice`: (Optional) Audio inputs for voice cloning.
    *   `lora`: (Optional) LoRA configuration.
    *   `temperature`: (Optional) Sampling temperature.
    *   `top_p`: (Optional) Nucleus sampling parameter.
    *   `voice_speed_factor`: (Optional) Speech rate adjustment for all speakers (0.8-1.2).

### 4.  VibeVoice Free Memory

*   Manually frees VibeVoice models from memory.
*   **Input**: `audio` - Triggered by audio input.
*   **Output**: `audio` - Passes through the input audio.
*   **Use Case**: Insert between nodes to free VRAM/RAM.

### 5.  VibeVoice LoRA

*   Configure and load custom LoRA adapters for fine-tuned VibeVoice models.
*   **LoRA Location**:  `ComfyUI/models/vibevoice/loras/`
*   **Parameters**:
    *   `lora_name`: Select from available LoRA adapters or "None" to disable.
    *   `llm_strength`: Strength of the language model LoRA (0.0-2.0).
    *   `use_llm`: Apply language model LoRA component.
    *   `use_diffusion_head`: Apply diffusion head replacement.
    *   `use_acoustic_connector`: Apply acoustic connector LoRA.
    *   `use_semantic_connector`: Apply semantic connector LoRA.
*   **Output**: `lora` -  LoRA configuration to connect to speaker nodes.

## Multi-Speaker Text Format

Use `[N]:` notation for multi-speaker conversations:

```
[1]: Hello, how are you today?
[2]: I'm doing great, thanks for asking!
[1]: That's wonderful to hear.
[3]: Hey everyone, mind if I join the conversation?
[2]: Not at all, welcome!
```

*   Use `[1]:`, `[2]:`, `[3]:`, `[4]:` for speaker labels.
*   Maximum 4 speakers supported.
*   The system auto-detects the number of speakers.
*   Each speaker can have an optional voice sample for cloning.

## Model Information

### VibeVoice-1.5B

*   **Size**: \~5GB
*   **Speed**: Faster inference.
*   **Quality**: Good for single speaker.
*   **Use Case**: Quick prototyping, single voices.

### VibeVoice-Large

*   **Size**: \~17GB
*   **Speed**: Slower inference.
*   **Quality**: Best quality.
*   **Use Case**: Highest quality, multi-speaker conversations.
*   **Note**: Latest official release from Microsoft.

### VibeVoice-Large-Quant-4Bit

*   **Size**: \~7GB
*   **Speed**: Balanced inference.
*   **Quality**: Good quality.
*   **Use Case**: Good quality, less VRAM, multi-speaker.
*   **Note**: Quantized by DevParker.

*Models are automatically downloaded on first use and cached in* `ComfyUI/models/vibevoice/`.

## Generation Modes

### Deterministic Mode (Default)

*   `use_sampling = False`
*   Consistent, stable output.
*   Recommended for production.

### Sampling Mode

*   `use_sampling = True`
*   More output variation.
*   Uses temperature and top\_p.
*   Good for creative exploration.

## Voice Cloning

1.  Connect an audio node to the `voice_to_clone` input (single speaker).
2.  Or connect to `speaker1_voice`, `speaker2_voice`, etc. (multi-speaker).
3.  The model attempts to match the voice characteristics.

**Requirements for Voice Samples:**

*   Clear audio, minimal noise.
*   Minimum 3-10 seconds. 30+ seconds recommended.
*   Automatically resampled to 24kHz.

## LoRA Support

### Overview

VibeVoice ComfyUI supports LoRA adapters for fine-tuning voice characteristics. Train and use specialized voice models.

### Setting Up LoRA Adapters

1.  **LoRA Directory Structure**:
    Place your LoRA adapter folders in: `ComfyUI/models/vibevoice/loras/`
    ```
    ComfyUI/
    └── models/
        └── vibevoice/
            └── loras/
                ├── my_custom_voice/
                │   ├── adapter_config.json
                │   ├── adapter_model.safetensors
                │   └── diffusion_head/  (optional)
                ├── character_voice/
                └── style_adaptation/
    ```

2.  **Required Files**:
    *   `adapter_config.json`: LoRA configuration.
    *   `adapter_model.safetensors` or `adapter_model.bin`: Model weights.
    *   Optional components:
        *   `diffusion_head/`: Custom diffusion head weights.
        *   `acoustic_connector/`: Acoustic connector adaptation.
        *   `semantic_connector/`: Semantic connector adaptation.

### Using LoRA in ComfyUI

1.  **Add VibeVoice LoRA Node**: Create a "VibeVoice LoRA" node and select your LoRA from the dropdown. Configure component settings and strength.
2.  **Connect to Speaker Nodes**: Connect the LoRA node's output to speaker nodes' `lora` input.
3.  **LoRA Parameters**:
    *   **llm\_strength**: Influence of the language model LoRA.
    *   **Component toggles**: Enable/disable specific LoRA components.
    *   Select "None" to disable LoRA.

### Training Your Own LoRA

Use the official fine-tuning repository:

*   **Repository**: [VibeVoice Fine-tuning](https://github.com/voicepowered-ai/VibeVoice-finetuning)
*   **Features**: Parameter-efficient fine-tuning, custom datasets, adjustable LoRA rank and scaling, optional diffusion head adaptation.

### Best Practices

*   **Voice Consistency**: Use the same LoRA across chunks.
*   **Memory Management**: LoRA adds minimal overhead.
*   **Compatibility**: Compatible with all VibeVoice models.
*   **Strength Tuning**: Start with default (1.0) and adjust.

### Compatibility Note

⚠️  **Transformers Version**:  The LoRA implementation was tested with `transformers==4.51.3`. For LoRA functionality with newer versions of transformers, use `transformers==4.51.3`:

```bash
pip install transformers==4.51.3
```

### 🙏 Credits

LoRA implementation by [@jpgallegoar](https://github.com/jpgallegoar) (PR #127)

## Voice Speed Control

### Overview

Influence speech rate by adjusting the speed of the reference voice.

**Available from version 1.5.0**

### How It Works

Applies time-stretching to the reference audio:
- Values < 1.0: slow down the reference voice
- Values > 1.0: speed up the reference voice

### Usage

- **Parameter**: `voice_speed_factor`
- **Range**: 0.8 to 1.2
- **Default**: 1.0 (normal)
- **Step**: 0.01 (1% increments)

### Recommended Settings

-   **Optimal Range**: 0.95 to 1.05
-   **Slower Speech**: Try 0.95 or 0.97
-   **Faster Speech**: Try 1.03 or 1.05
-   **Best Results**: Use 20+ second reference audio

### Important Notes

- Works best with longer reference audio.
- Extreme values may sound unnatural.
- In Multi Speaker mode, the adjustment applies to all speakers.
- Synthetic voices are unaffected.

### 📖 Examples

```
# Single Speaker
voice_speed_factor: 0.95  # Slightly slower speech
voice_speed_factor: 1.05  # Slightly faster speech

# Multi Speaker
voice_speed_factor: 0.98  # All speakers talk 2% slower
voice_speed_factor: 1.02  # All speakers talk 2% faster
```

## Pause Tags Support

### Overview

Insert silences between text segments.  **Not a standard VibeVoice feature**, but an original implementation.

**Available from version 1.3.0**

### Usage

*   `[pause]` - 1-second silence (default).
*   `[pause:ms]` - Custom duration in milliseconds (e.g., `[pause:2000]` for 2 seconds).

### 📖 Examples

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

### How It Works

1.  Wrapper parses text for pause tags.
2.  Segments between pauses are processed independently.
3.  Silence audio generated for each pause.
4.  Audio segments are concatenated.

### Best Practices

*   Use pauses at natural breaks.
*   Avoid pauses in the middle of phrases.
*   Test different durations.

## Tips for Best Results

1.  **Text Preparation**: Proper punctuation, break long texts, clear speaker transitions, use pause tags sparingly.
2.  **Model Selection**: 1.5B for speed, Large for quality/multi-speaker, Large-Quant-4Bit for low VRAM.
3.  **Seed Management**: Save good seeds for character voices, try random seeds.
4.  **Performance**: GPU recommended.

## System Requirements

### Hardware

*   **Minimum**: 8GB VRAM for VibeVoice-1.5B.
*   **Recommended**: 17GB+ VRAM for VibeVoice-Large.
*   **RAM**: 16GB+ system memory.

### Software

*   Python 3.8+.
*   PyTorch 2.0+.
*   CUDA 11.8+ (for GPU).
*   Transformers 4.51.3+.
*   ComfyUI (latest).

## Troubleshooting

### Installation Issues

*   Use ComfyUI's Python environment.
*   Try manual installation.
*   Restart ComfyUI.

### Generation Issues

*   Use deterministic mode for voice stability.
*   Ensure correct multi-speaker format.
*   Check speaker numbers are sequential.

### Memory Issues

*   Use the 1.5B model.
*   Use VRAM cleanup.

## Examples

### Single Speaker

```
Text: "Welcome...intelligence."
Model: VibeVoice-1.5B
cfg_scale: 1.3
use_sampling: False
```

### Two Speakers

```
[1]:... developments?
[2]:... impressive!
```

### Four Speaker Conversation

```
[1]:... meeting.
[2]:... us!
[3]:... here.
[4]:... discussion.
```

## Performance Benchmarks

| Model                   | VRAM Usage | Context Length | Max Audio Duration |
| ----------------------- | ---------- | ---------------- | ------------------ |
| VibeVoice-1.5B          | ~8GB       | 64K tokens       | \~90 minutes      |
| VibeVoice-Large         | ~17GB      | 32K tokens       | \~45 minutes      |
| VibeVoice-Large-Quant-4Bit | ~7GB       | 32K tokens       | \~45 minutes      |

## Known Limitations

*   Maximum 4 speakers.
*   Best for English and Chinese.
*   Some seeds may produce unstable output.
*   No background music control.

## License

MIT License. See the `LICENSE` file for details.

**Note**:  VibeVoice model subject to Microsoft's licensing terms.

## Links

*   [Original VibeVoice Repository](https://github.com/microsoft/VibeVoice) - Official Microsoft VibeVoice repository (currently unavailable)

## Credits

*   **VibeVoice Model**: Microsoft Research
*   **ComfyUI Integration**: Fabio Sarracino
*   **Base Model**: Built on Qwen2.5 architecture

## Support

1.  Check the troubleshooting section.
2.  Review ComfyUI logs.
3.  Ensure VibeVoice is installed.
4.  Open an issue with details.

## Contributing

1.  Test changes.
2.  Follow code style.
3.  Update documentation.
4.  Submit pull requests.

## Changelog

### Version 1.5.0

*   Added Voice Speed Control.

### Version 1.4.3

*   Improved LoRA system.

### Version 1.4.2

*   Bug Fixing

### Version 1.4.1

*   Fixed HuggingFace authentication error

### Version 1.4.0

*   Added LoRA support.

### Version 1.3.0

*   Added custom pause tag support.

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

*   Added interruption handler
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
*   Single speaker node with voice cloning
*   Multi-speaker node with automatic speaker detection
*   Text file loading from ComfyUI directories
*   Deterministic and sampling generation modes
*   Support for VibeVoice 1.5B and Large models