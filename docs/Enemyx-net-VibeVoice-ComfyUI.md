# VibeVoice ComfyUI Integration: Turn Text into Stunning Speech

Unleash the power of Microsoft's VibeVoice directly within ComfyUI, generating realistic single and multi-speaker audio with ease. [Original Repo](https://github.com/Enemyx-net/VibeVoice-ComfyUI)

## Key Features:

*   🎤 **High-Quality TTS:** Generate natural-sounding speech from text.
*   🗣️ **Multi-Speaker Support:** Create dynamic conversations with up to 4 voices.
*   🔊 **Voice Cloning:** Replicate voices from audio samples.
*   📝 **Text File Loading:** Import scripts directly from text files.
*   ⚙️ **Flexible Customization:** Control temperature, sampling, and guidance.
*   🚀 **Model Options:** Choose from various models optimized for speed and quality.
*   🍎 **Apple Silicon Support:** Native GPU acceleration on M1/M2/M3 Macs via MPS
*   ⚡ **Optimized Performance:** Choose different Attention Mechanisms for faster generation
*   🧠 **Memory Management:** Free VRAM with dedicated nodes for complex workflows.

## Getting Started

### Automatic Installation (Recommended)

1.  Clone the repository into your ComfyUI custom nodes folder:

    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/Enemyx-net/VibeVoice-ComfyUI
    ```

2.  Restart ComfyUI - requirements will automatically install on first use.

## Available Nodes

### 1.  VibeVoice Load Text From File

*   Loads text content from .txt files.
*   **Output:** Text string.

### 2.  VibeVoice Single Speaker

*   Generates speech from text with a single voice.
    *   **Parameters:**
        *   `text`: Input text.
        *   `model`: VibeVoice-1.5B, VibeVoice-Large, or VibeVoice-Large-Quant-4Bit.
        *   `attention_type`: (default: auto) auto, eager, sdpa, flash\_attention\_2 or sage
        *   `free_memory_after_generate`: (default: True)
        *   `diffusion_steps`: Number of denoising steps (default: 20).
        *   `seed`: Random seed (default: 42).
        *   `cfg_scale`: Classifier-free guidance (default: 1.3).
        *   `use_sampling`: Enable/disable deterministic generation (default: False)
        *   Optional parameters: `voice_to_clone`, `temperature`, `top_p`, `max_words_per_chunk`
*   **Voice Cloning:** Optional audio input for voice cloning.

### 3.  VibeVoice Multiple Speakers

*   Generates multi-speaker conversations.
    *   **Speaker Format:**  Use `[N]:` notation (N = 1-4).
    *   **Parameters:**
        *   `text`: Input text with speaker labels.
        *   `model`: VibeVoice-1.5B, VibeVoice-Large, or VibeVoice-Large-Quant-4Bit.
        *   `attention_type`: (default: auto) auto, eager, sdpa, flash\_attention\_2 or sage
        *   `free_memory_after_generate`: (default: True)
        *   `diffusion_steps`: Number of denoising steps (default: 20).
        *   `seed`: Random seed (default: 42).
        *   `cfg_scale`: Classifier-free guidance (default: 1.3).
        *   `use_sampling`: Enable/disable deterministic generation (default: False)
        *   Optional parameters: `speaker1_voice` to `speaker4_voice`, `temperature`, `top_p`.
*   **Voice Assignment:** Optional voice samples for each speaker.

### 4.  VibeVoice Free Memory

*   Manually frees VibeVoice models from memory.
*   **Input:** `audio`
*   **Output:** `audio`

## Advanced Features

### Multi-Speaker Text Format

*   Use `[N]:` (1-4) to label speakers:

    ```
    [1]: Hello, how are you today?
    [2]: I'm doing great!
    ```

### Model Information

*   **VibeVoice-1.5B:** Fast, lower memory (~5GB).
*   **VibeVoice-Large:** Highest quality (~17GB).
*   **VibeVoice-Large-Quant-4Bit:** Balanced quality and memory usage (~7GB).

### Generation Modes

*   **Deterministic (Default):** Consistent output (`use_sampling = False`).
*   **Sampling:**  More varied output (`use_sampling = True`), use temperature and top\_p.

### Voice Cloning

1.  Connect an audio node to `voice_to_clone` (single speaker) or `speakerN_voice` (multi-speaker).
2.  Requires clear audio (3-30 seconds, minimal noise).

### Pause Tags Support

*   Insert silences using:
    *   `[pause]` (1-second)
    *   `[pause:ms]` (custom duration in milliseconds)
*   **Warning:** Pauses can split context, affecting speech flow.

## Tips for Optimal Results

*   Prepare text with punctuation and breaks.
*   Select the appropriate model based on your needs.
*   Save good seeds for character consistency.
*   Use a GPU for faster generation.

## System Requirements

*   **Minimum:** 8GB VRAM.
*   **Recommended:** 17GB+ VRAM, 16GB+ RAM.
*   **Software:** Python 3.8+, PyTorch 2.0+, CUDA 11.8+ (for GPU), Transformers 4.51.3+, ComfyUI.

## Troubleshooting and Resources

*   Check the Troubleshooting section for common issues.
*   Review ComfyUI logs for error messages.
*   [Original VibeVoice Repository](https://github.com/microsoft/VibeVoice) - Note: Official Microsoft repository (currently unavailable)

## Changelog

*   **Version 1.3.0:** Added Custom pause tag support
*   **Version 1.2.5:** Bug Fixing
*   **Version 1.2.4:** Added automatic text chunking
*   **Version 1.2.3:** Added SageAttention support
*   **Version 1.2.2:** Added 4-bit quantized model support
*   **Version 1.2.1:** Bug Fixing
*   **Version 1.2.0:** MPS Support for Apple Silicon
*   **Version 1.1.1:** Universal Transformers Compatibility
*   **Version 1.1.0:** Updated the URL for downloading the VibeVoice-Large model
*   **Version 1.0.9:** Embedded VibeVoice code
*   **Version 1.0.8:** BFloat16 Compatibility Fix
*   **Version 1.0.7:** Added interruption handler
*   **Version 1.0.6:** Fixed a bug that prevented VibeVoice nodes from receiving audio directly from another VibeVoice node
*   **Version 1.0.5:** Added support for Microsoft's official VibeVoice-Large model
*   **Version 1.0.4:** Improved tokenizer dependency handling
*   **Version 1.0.3:** Added `attention_type` & `diffusion_steps` parameters
*   **Version 1.0.2:** Added `free_memory_after_generate` toggle & New dedicated "Free Memory Node"
*   **Version 1.0.1:** Fixed issue with line breaks in speaker text
*   **Version 1.0.0:** Initial release

## License

MIT License.  See the LICENSE file.