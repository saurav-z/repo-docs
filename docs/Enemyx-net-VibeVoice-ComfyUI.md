# VibeVoice ComfyUI Nodes: Transform Text into Realistic Speech

Unleash the power of Microsoft's VibeVoice directly within ComfyUI! Generate stunning, high-quality text-to-speech with unparalleled realism. [Visit the original repository](https://github.com/Enemyx-net/VibeVoice-ComfyUI) for more details.

## Key Features

*   🎤 **Single & Multi-Speaker TTS:** Effortlessly create single and multi-voice audio.
*   🗣️ **Voice Cloning:** Clone voices from audio samples for personalized audio.
*   📚 **Text File Support:** Load and process text from files directly within ComfyUI.
*   🗣️ **Flexible Speaker Control:** Utilize up to 4 distinct speakers in conversations.
*   ⏱️ **Custom Pause Tags:** Precisely control speech pacing with customizable pauses.
*   ⚡ **Optimized Performance:** Benefit from various attention mechanisms & memory management.
*   🍎 **Apple Silicon Support:** Native GPU acceleration for M1/M2/M3 Macs.
*   💾 **4-Bit Quantization:** Reduced memory usage with minimal quality loss.

## Core Functionality & Node Details

### 1. VibeVoice Load Text From File

*   **Purpose:** Load text content from files.
*   **Supported Formats:** `.txt`
*   **Output:** Text string for downstream TTS nodes.

### 2. VibeVoice Single Speaker

*   **Purpose:** Generates speech using a single voice.
*   **Inputs:** Text (direct or from "Load Text" node), optional voice cloning audio.
*   **Models:** VibeVoice-1.5B, VibeVoice-Large, VibeVoice-Large-Quant-4Bit
*   **Parameters**: `text`, `model`, `attention_type`, `free_memory_after_generate`, `diffusion_steps`, `seed`, `cfg_scale`, `use_sampling`, `voice_to_clone`, `temperature`, `top_p`, `max_words_per_chunk`
    *   **Voice Cloning:** Connect an audio node to `voice_to_clone`. Requires clean audio (3-30+ seconds).
*   **Generation Modes:**
    *   **Deterministic Mode:** `use_sampling = False` (consistent output).
    *   **Sampling Mode:** `use_sampling = True` (more variation, uses `temperature` and `top_p`).

### 3. VibeVoice Multiple Speakers

*   **Purpose:** Generates multi-speaker conversations.
*   **Inputs:** Text (with speaker labels using `[N]:` notation, where N is 1-4), optional voice cloning audio for each speaker.
*   **Recommended:** VibeVoice-Large for best multi-speaker quality.
*   **Parameters**: `text`, `model`, `attention_type`, `free_memory_after_generate`, `diffusion_steps`, `seed`, `cfg_scale`, `use_sampling`, `speaker1_voice` to `speaker4_voice`, `temperature`, `top_p`
    *   **Voice Assignment**: Connect to `speaker1_voice`, `speaker2_voice`, etc. for voice cloning.

#### Multi-Speaker Text Format

```
[1]: Hello, how are you today?
[2]: I'm doing great, thanks for asking!
[1]: That's wonderful to hear.
[3]: Hey everyone, mind if I join the conversation?
[2]: Not at all, welcome!
```

### 4. VibeVoice Free Memory

*   **Purpose:** Manually releases VRAM/RAM.
*   **Input:** Audio (connect audio output from other nodes to trigger cleanup).
*   **Output:** Passes through the input audio unchanged.
*   **Usage:** Insert between nodes to manage memory in complex workflows.

## Model Information

### VibeVoice-1.5B

*   **Size:** ~5GB
*   **Speed:** Faster inference
*   **Quality:** Good for single speaker
*   **Use Case:** Prototyping, single voices.

### VibeVoice-Large

*   **Size:** ~17GB
*   **Speed:** Slower, optimized inference
*   **Quality:** Best available
*   **Use Case:** Highest quality production, multi-speaker.

### VibeVoice-Large-Quant-4Bit

*   **Size:** ~7GB
*   **Speed:** Balanced inference
*   **Quality:** Good quality
*   **Use Case:** Good quality with less VRAM, multi-speaker.

Models are automatically downloaded and cached in `ComfyUI/models/vibevoice/`.

## Pause Tags Support (Wrapper Feature)

Insert silences between text segments for better pacing.
*   `[pause]` - 1-second silence.
*   `[pause:ms]` - Custom duration in milliseconds.

### Usage
```
[1]: Hello everyone [pause] how are you doing today?
[2]: I'm doing great! [pause:500] Thanks for asking.
[1]: Wonderful to hear!
```

> **⚠️Note:** Pause tags split the text into chunks, potentially affecting context. Use sparingly.

## Installation

### Automatic Installation (Recommended)

1.  Clone the repository into your ComfyUI custom nodes folder:

    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/Enemyx-net/VibeVoice-ComfyUI
    ```

2.  Restart ComfyUI. Requirements will install automatically on first use.

## Tips for Best Results

1.  **Text Preparation:** Use punctuation, break up long texts, and format multi-speaker text correctly.
2.  **Model Selection:** Choose models based on your needs (speed, quality, and VRAM).
3.  **Seed Management:** Save good seeds for consistent voices.
4.  **Performance:** Use GPU for faster processing, and use `free_memory_after_generate = True` to free up VRAM

## System Requirements

*   **Hardware:**
    *   **Minimum:** 8GB VRAM (for VibeVoice-1.5B)
    *   **Recommended:** 17GB+ VRAM (for VibeVoice-Large)
    *   **RAM:** 16GB+
*   **Software:**
    *   Python 3.8+
    *   PyTorch 2.0+
    *   CUDA 11.8+ (for GPU acceleration)
    *   Transformers 4.51.3+
    *   ComfyUI (latest version)

## Troubleshooting

*   **Installation:** Ensure you're in the ComfyUI environment, try manual installation if automatic fails, and restart ComfyUI.
*   **Generation:** Use deterministic mode if voices sound unstable, check multi-speaker text formatting, and verify speaker numbers.
*   **Memory:** Consider the model size (use smaller models or the "Free Memory" node).

## Examples

```
# Single Speaker
Text: "Welcome to our presentation. Today we'll explore the fascinating world of artificial intelligence."
Model: VibeVoice-1.5B
cfg_scale: 1.3
use_sampling: False
```

```
# Two Speakers
[1]: Have you seen the new AI developments?
[2]: Yes, they're quite impressive!
[1]: I think voice synthesis has come a long way.
[2]: Absolutely, it sounds so natural now.
```

## Performance Benchmarks

| Model                    | VRAM Usage | Context Length | Max Audio Duration |
|--------------------------|------------|----------------|-------------------|
| VibeVoice-1.5B           | ~8GB | 64K tokens | ~90 minutes |
| VibeVoice-Large          | ~17GB | 32K tokens | ~45 minutes |
| VibeVoice-Large-Quant-4Bit | ~7GB | 32K tokens | ~45 minutes |

## Known Limitations

*   Maximum 4 speakers in multi-speaker mode.
*   Best results with English and Chinese text.
*   Some seeds may cause unstable output.
*   Background music generation is not directly controllable.

## License

This ComfyUI wrapper is released under the MIT License. The VibeVoice model itself is subject to Microsoft's licensing terms.

## Credits

*   **VibeVoice Model:** Microsoft Research
*   **ComfyUI Integration:** Fabio Sarracino

## Changelog

A detailed changelog is available in the original README.