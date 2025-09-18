# VibeVoice ComfyUI: Transform Text into Natural Speech with Powerful AI

Unleash the power of Microsoft's VibeVoice text-to-speech directly within ComfyUI workflows, creating realistic single and multi-speaker audio with ease.  [Original Repo](https://github.com/Enemyx-net/VibeVoice-ComfyUI)

## Key Features

*   🎤 **Single Speaker TTS:** Generate natural speech with optional voice cloning.
*   👥 **Multi-Speaker Conversations:** Support for up to 4 distinct speakers, create engaging dialogues.
*   🗣️ **Voice Cloning:** Clone voices from audio samples for personalized speech.
*   📝 **Text File Loading:** Load scripts directly from text files.
*   💾 **Automatic Text Chunking:** Handles long texts seamlessly.
*   ⏱️ **Custom Pause Tags:** Fine-tune speech with  `[pause]` and `[pause:ms]` tags for precise control.
*   🔄 **Node Chaining:** Connect multiple nodes for complex audio creation workflows.
*   ⚡ **Performance Optimization:** Choose between multiple attention mechanisms and adjust diffusion steps.
*   🍎 **Apple Silicon Support:** Native GPU acceleration on M1/M2/M3 Macs via MPS.
*   💾 **Memory Management:**  Optimize VRAM usage, manual memory control.
*   🚀 **Multiple Model Variants:** Choose from VibeVoice-1.5B, VibeVoice-Large, and VibeVoice-Large-Quant-4Bit for speed, quality, or balance.

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

### 1. VibeVoice Load Text From File

*   Loads text content from .txt files.
*   **Output:** Text string for TTS nodes.

### 2. VibeVoice Single Speaker

*   Generates speech from text using a single voice.
*   **Text Input:** Direct text or connection from Load Text node.
*   **Models:** VibeVoice-1.5B, VibeVoice-Large or VibeVoice-Large-Quant-4Bit.
*   **Voice Cloning:** Optional audio input.
*   **Parameters:**  `text`, `model`, `attention_type`, `free_memory_after_generate`, `diffusion_steps`, `seed`, `cfg_scale`, `use_sampling`, `voice_to_clone`, `temperature`, `top_p`, `max_words_per_chunk`.

### 3. VibeVoice Multiple Speakers

*   Generates multi-speaker conversations.
*   **Speaker Format:** Use `[N]:` (1-4).
*   **Voice Assignment:** Optional audio input for each speaker.
*   **Recommended Model:** VibeVoice-Large.
*   **Parameters:** `text`, `model`, `attention_type`, `free_memory_after_generate`, `diffusion_steps`, `seed`, `cfg_scale`, `use_sampling`,  `speaker1_voice` - `speaker4_voice`, `temperature`, `top_p`.

### 4. VibeVoice Free Memory

*   Manually frees VibeVoice models.
*   **Input:** `audio`
*   **Output:** `audio`
*   **Use Case:** Free VRAM/RAM in complex workflows.

## Multi-Speaker Text Format

```
[1]: Hello, how are you today?
[2]: I'm doing great, thanks for asking!
[1]: That's wonderful to hear.
[3]: Hey everyone, mind if I join the conversation?
[2]: Not at all, welcome!
```

*   Use `[1]:`, `[2]:`, `[3]:`, `[4]:` for speaker labels (max 4 speakers).
*   Each speaker can have an optional voice sample for cloning.

## Model Information

### VibeVoice-1.5B

*   **Size:** \~5GB.
*   **Speed:** Faster inference.
*   **Quality:** Good for single speaker.
*   **Use Case:** Quick prototyping, single voices.

### VibeVoice-Large

*   **Size:** \~17GB.
*   **Speed:** Slower inference, best quality.
*   **Quality:** Best available quality.
*   **Use Case:** Highest quality production, multi-speaker conversations.
*   **Note**: Latest official release from Microsoft.

### VibeVoice-Large-Quant-4Bit

*   **Size**: \~7GB.
*   **Speed**: Balanced inference.
*   **Quality**: Good quality.
*   **Use Case**: Good quality production with less VRAM, multi-speaker conversations.
*   **Note**: Quantized by DevParker.

Models are automatically downloaded on first use and cached in `ComfyUI/models/vibevoice/`.

## Generation Modes

### Deterministic Mode (Default)

*   `use_sampling = False`
*   Consistent, stable output.
*   Recommended for production.

### Sampling Mode

*   `use_sampling = True`
*   More variation.
*   Uses `temperature` and `top_p`.
*   Good for creative exploration.

## Voice Cloning

1.  Connect an audio node to `voice_to_clone` (single speaker).
2.  Or connect to `speaker1_voice`, etc. (multi-speaker).

**Requirements for voice samples:**

*   Clear audio with minimal noise.
*   Minimum 3-10 seconds (30 seconds recommended).
*   Automatically resampled to 24kHz.

## Pause Tags Support

*   Use pause tags to control speech pacing.
*   **Available from version 1.3.0**
*   **Note: The pause forces the text to be split into chunks. This may worsen the model's ability to understand the context. The model's context is represented ONLY by its own chunk.**
    *   Text before a pause and text after a pause are processed separately
    *   The model cannot see across pause boundaries when generating speech
    *   This may affect prosody and intonation consistency
    *   Use pauses sparingly for best results

*   `[pause]` - Inserts a 1-second silence.
*   `[pause:ms]` - Inserts a custom silence in milliseconds.

### Examples
```
Welcome to our presentation. [pause] Today we'll explore artificial intelligence. [pause:500] Let's begin!
```
```
[1]: Hello everyone [pause] how are you doing today?
[2]: I'm doing great! [pause:500] Thanks for asking.
[1]: Wonderful to hear!
```

## Tips for Best Results

1.  **Text Preparation**:
    *   Use proper punctuation.
    *   Break long texts into paragraphs.
    *   For multi-speaker, ensure clear transitions.
    *   Use pause tags sparingly to maintain context continuity.

2.  **Model Selection**:
    *   Use 1.5B for quick single-speaker tasks.
    *   Use Large for best quality and multi-speaker.
    *   Use Large-Quant-4Bit for good quality and low VRAM usage.

3.  **Seed Management**:
    *   Default seed (42) works well.
    *   Save good seeds.
    *   Try random seeds.

4.  **Performance**:
    *   First run downloads models.
    *   Subsequent runs use cached models.
    *   GPU recommended.

## System Requirements

### Hardware

*   **Minimum**: 8GB VRAM (VibeVoice-1.5B).
*   **Recommended**: 17GB+ VRAM (VibeVoice-Large).
*   **RAM**: 16GB+ system memory.

### Software

*   Python 3.8+.
*   PyTorch 2.0+.
*   CUDA 11.8+ (for GPU acceleration).
*   Transformers 4.51.3+.
*   ComfyUI (latest version).

## Troubleshooting

*   Check the Troubleshooting section.
*   Review ComfyUI logs for errors.
*   Ensure VibeVoice is properly installed.
*   Open an issue with detailed error information.

## Examples

```
Text: "Welcome to our presentation. Today we'll explore the fascinating world of artificial intelligence."
Model: VibeVoice-1.5B
cfg_scale: 1.3
use_sampling: False
```

```
[1]: Have you seen the new AI developments?
[2]: Yes, they're quite impressive!
[1]: I think voice synthesis has come a long way.
[2]: Absolutely, it sounds so natural now.
```

```
[1]: Welcome everyone to our meeting.
[2]: Thanks for having us!
[3]: Glad to be here.
[4]: Looking forward to the discussion.
[1]: Let's begin with the agenda.
```

## Performance Benchmarks

| Model                      | VRAM Usage | Context Length | Max Audio Duration |
| -------------------------- | ---------- | -------------- | ------------------ |
| VibeVoice-1.5B             | \~8GB      | 64K tokens     | \~90 minutes       |
| VibeVoice-Large            | \~17GB     | 32K tokens     | \~45 minutes       |
| VibeVoice-Large-Quant-4Bit | \~7GB      | 32K tokens     | \~45 minutes       |

## Known Limitations

*   Maximum 4 speakers.
*   Best with English and Chinese.
*   Some seeds may cause instability.
*   Limited background music control.

## License

Released under the MIT License. See LICENSE file.

**Note**: VibeVoice model is subject to Microsoft's licensing terms. Check their repository for details.

## Links

*   [Original VibeVoice Repository](https://github.com/microsoft/VibeVoice)

## Credits

*   **VibeVoice Model**: Microsoft Research.
*   **ComfyUI Integration**: Fabio Sarracino.
*   **Base Model**: Built on Qwen2.5 architecture.

## Support

For issues or questions, check the troubleshooting section, review ComfyUI logs, or open an issue.

## Contributing

Contributions welcome!  Follow the guidelines, test changes, and submit pull requests.

## Changelog

*(See original README for the changelog)*