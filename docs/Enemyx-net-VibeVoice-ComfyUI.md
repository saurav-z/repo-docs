# VibeVoice ComfyUI: Effortlessly Generate High-Quality Speech within ComfyUI

This ComfyUI integration brings Microsoft's VibeVoice text-to-speech model directly into your workflows, enabling stunning, natural-sounding audio generation with features like multi-speaker conversations and voice cloning. [Go to the original repository](https://github.com/Enemyx-net/VibeVoice-ComfyUI) to get started!

**Key Features:**

*   **Single & Multi-Speaker TTS:** Create realistic speech with up to 4 distinct voices.
*   **Voice Cloning:** Clone voices from audio samples for personalized audio.
*   **Model Options:** Choose from optimized models for speed, quality, and memory efficiency.
*   **Flexible Configuration:** Control parameters like temperature, sampling, and guidance scale.
*   **Memory Management:** Optimize VRAM usage with automatic and manual control.
*   **Apple Silicon Support:** Native GPU acceleration on M1/M2/M3 Macs via MPS.
*   **Pause Tag Support**: Enhance speech pacing with custom pause tags.

## Quick Start

### Installation

**Automatic Installation (Recommended):**

1.  Navigate to your ComfyUI custom nodes directory:
    ```bash
    cd ComfyUI/custom_nodes
    ```
2.  Clone the repository:
    ```bash
    git clone https://github.com/Enemyx-net/VibeVoice-ComfyUI
    ```
3.  Restart ComfyUI. Dependencies will install automatically on first use.

## Nodes Overview

### 1. VibeVoice Load Text From File
*   **Functionality:** Loads text from `.txt` files.
*   **Output:** Text string.

### 2. VibeVoice Single Speaker
*   **Functionality:** Generates speech from text with a single voice, including voice cloning.
*   **Parameters:**
    *   `text`: Input text.
    *   `model`: Choose from `VibeVoice-1.5B`, `VibeVoice-Large`, or `VibeVoice-Large-Quant-4Bit`.
    *   Other parameters: `attention_type`, `free_memory_after_generate`, `diffusion_steps`, `seed`, `cfg_scale`, `use_sampling`.
    *   Optional:  `voice_to_clone`, `temperature`, `top_p`, `max_words_per_chunk`.

### 3. VibeVoice Multiple Speakers
*   **Functionality:** Generates multi-speaker conversations.
*   **Format:** Use `[N]:` where N is the speaker number (1-4).
*   **Parameters:** (Similar to Single Speaker, includes `speaker1_voice` through `speaker4_voice` for voice cloning)

### 4. VibeVoice Free Memory
*   **Functionality:** Frees VRAM used by VibeVoice models.
*   **Input:** `audio` (connect audio output to trigger).
*   **Output:** `audio`.

## Model Information

*   **VibeVoice-1.5B:** ~5GB, fast inference, good for single speaker.
*   **VibeVoice-Large:** ~17GB, best quality, optimized for multi-speaker.
*   **VibeVoice-Large-Quant-4Bit:** ~7GB, good quality, lower VRAM usage.

## Additional Features

*   **Generation Modes:** Deterministic (default, stable output) and Sampling (more variation).
*   **Voice Cloning:** Connect audio samples to `voice_to_clone` or `speakerN_voice` inputs.
*   **Pause Tags:** Use `[pause]` (1-second silence) and `[pause:ms]` tags within text for custom pacing (wrapper feature).

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

## Troubleshooting

*   Ensure ComfyUI's Python environment is used.
*   Restart ComfyUI after installation.
*   Check speaker formats for multi-speaker issues.

## Other Important Information
* **System Requirements** See original README for hardware and software requirements
* **Limitations** Maximum 4 speakers in multi-speaker mode
* **Credits**: Microsoft Research, Fabio Sarracino, and DevParker