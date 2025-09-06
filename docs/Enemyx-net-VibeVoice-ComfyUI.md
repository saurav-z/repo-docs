# VibeVoice ComfyUI: Unleash High-Quality Text-to-Speech in Your Workflows

**Transform text into lifelike speech with VibeVoice ComfyUI, a seamless integration for ComfyUI that brings the power of Microsoft's VibeVoice models directly to your creative workflows.** ([Original Repo](https://github.com/Enemyx-net/VibeVoice-ComfyUI))

## Key Features

*   🎤 **Single-Speaker TTS:** Generate natural-sounding speech with optional voice cloning from audio samples.
*   👥 **Multi-Speaker Conversations:** Create dynamic dialogues with up to four distinct speakers.
*   🔊 **Voice Cloning:** Replicate voices accurately from audio samples.
*   📝 **Text File Loading:** Easily import scripts from .txt files.
*   🚀 **Model Flexibility:** Choose between the faster VibeVoice-1.5B and the higher-quality VibeVoice-Large models.
*   ⚙️ **Customization Options:** Fine-tune your output with adjustable temperature, sampling, guidance scale, and more.
*   ⚡ **Performance Optimization:** Utilize attention mechanisms (auto, eager, sdpa, flash\_attention\_2), adjustable diffusion steps, memory management tools, and manual memory control for optimal performance.

## Getting Started

### Installation (Automatic, Recommended)

1.  Navigate to your ComfyUI custom nodes folder:
    ```bash
    cd ComfyUI/custom_nodes
    ```
2.  Clone the repository:
    ```bash
    git clone https://github.com/Enemyx-net/VibeVoice-ComfyUI
    ```
3.  Restart ComfyUI. The necessary dependencies will be automatically installed upon first use.

## Available Nodes

### 1. VibeVoice Load Text From File

*   **Function:** Loads text from .txt files within ComfyUI's input/output/temp directories.
*   **Output:** Returns a text string for use with TTS nodes.

### 2. VibeVoice Single Speaker

*   **Function:** Converts text into speech using a single voice.
*   **Inputs:**
    *   `text`: Input text (direct or from Load Text node).
    *   `model`: VibeVoice-1.5B or VibeVoice-Large.
    *   `voice_to_clone` (Optional): Audio input for voice cloning.
    *   Other parameters for fine-tuning such as `attention_type`, `free_memory_after_generate`, `diffusion_steps`, `seed`, `cfg_scale`, `use_sampling`, `temperature`, and `top_p`.
*   **Outputs:** Audio output.

### 3. VibeVoice Multiple Speakers

*   **Function:** Generates multi-speaker conversations.
*   **Speaker Format:**  Uses `[N]:` where N is speaker number (1-4).
*   **Inputs:**
    *   `text`:  Text formatted with speaker labels.
    *   `model`: VibeVoice-1.5B or VibeVoice-Large (VibeVoice-Large recommended for optimal quality).
    *   `speaker[1-4]_voice` (Optional): Audio inputs for voice cloning for each speaker.
    *   Other parameters for fine-tuning such as `attention_type`, `free_memory_after_generate`, `diffusion_steps`, `seed`, `cfg_scale`, `use_sampling`, `temperature`, and `top_p`.
*   **Outputs:** Audio output.

### 4. VibeVoice Free Memory

*   **Function:** Manually frees loaded VibeVoice models from memory.
*   **Input:**  Connect an audio node to trigger memory cleanup.
*   **Output:** Passes the input audio unchanged.
*   **Use Case:** Optimize memory management in complex workflows.

## Multi-Speaker Text Formatting

Use the following format for multi-speaker generation:

```
[1]: Hello, how are you today?
[2]: I'm doing great, thanks!
```

*   Use `[1]:`, `[2]:`, `[3]:`, or `[4]:` for speaker labels.
*   Supports a maximum of four speakers.
*   Voice cloning can be applied to each speaker.

## Model Information

### VibeVoice-1.5B

*   **Size:** ~5GB download
*   **Speed:** Faster inference
*   **Quality:** Good for single-speaker tasks
*   **Use Case:** Quick prototyping, single-speaker voices.

### VibeVoice-Large

*   **Size:** ~17GB download
*   **Speed:** Slower inference, but optimized.
*   **Quality:** Highest quality output.
*   **Use Case:** Production-level quality and multi-speaker conversations.
*   **Note:** The latest official release from Microsoft.

*Models are automatically downloaded and cached in `ComfyUI/models/vibevoice/` on first use.*

## Generation Modes

### Deterministic Mode (Default)

*   `use_sampling = False`
*   Produces consistent output.
*   Recommended for production.

### Sampling Mode

*   `use_sampling = True`
*   Offers more output variation.
*   Uses `temperature` and `top_p` parameters.
*   Ideal for creative exploration.

## Voice Cloning

1.  Connect an audio node to the `voice_to_clone` input (Single Speaker) or `speaker[1-4]_voice` (Multi-Speaker).
2.  The model will attempt to match voice characteristics.

**Voice Sample Requirements:**

*   Clear audio with minimal background noise.
*   Minimum duration: 3-10 seconds.  30+ seconds recommended for best results.
*   Automatically resampled to 24kHz.

## Tips for Optimal Results

1.  **Text Preparation:** Utilize proper punctuation, and break long texts into paragraphs.
2.  **Model Selection:** Choose VibeVoice-1.5B for speed and single-speaker projects, and VibeVoice-Large for highest quality and multi-speaker use.
3.  **Seed Management:** Save successful seeds for consistent character voices; use random seeds for variety.
4.  **Performance:**  The first run downloads models. GPUs are recommended for faster inference.

## System Requirements

### Hardware

*   **Minimum:** 8GB VRAM for VibeVoice-1.5B, 16GB for VibeVoice-Large.
*   **Recommended:** 17GB+ VRAM for VibeVoice-Large for optimal performance.
*   **RAM:** 16GB+ system memory.

### Software

*   Python 3.8+
*   PyTorch 2.0+
*   CUDA 11.8+ (for GPU acceleration)
*   Transformers 4.51.3+
*   ComfyUI (latest version)

## Troubleshooting

*   **Installation:** Ensure you are using ComfyUI's Python environment, restart ComfyUI, and try manual installation if necessary.
*   **Generation:** For unstable voices, try deterministic mode. For multi-speaker, verify correct `[N]:` formatting.
*   **Memory:** Use the appropriate model based on VRAM availability. Utilize the Free Memory Node.

## Examples

### Single Speaker

```
Text: "Welcome to our presentation..."
Model: VibeVoice-1.5B
cfg_scale: 1.3
use_sampling: False
```

### Two Speakers

```
[1]: Have you seen the new AI developments?
[2]: Yes, they're quite impressive!
```

### Four Speaker Conversation

```
[1]: Welcome everyone to our meeting.
[2]: Thanks for having us!
[3]: Glad to be here.
[4]: Looking forward to the discussion.
```

## Performance Benchmarks (Approximate)

| Model                  | VRAM Usage | Context Length | Max Audio Duration |
|------------------------|------------|----------------|-------------------|
| VibeVoice-1.5B         | ~8GB | 64K tokens | ~90 minutes |
| VibeVoice-Large        | ~16GB      | 32K tokens      | ~45 minutes      |

## Known Limitations

*   Maximum 4 speakers in multi-speaker mode.
*   Best results with English and Chinese text.
*   Some seeds may produce unstable output.
*   Direct control of background music generation is not available.

## License

This ComfyUI wrapper is released under the MIT License. See the `LICENSE` file for more details.

**Note:** The VibeVoice model itself is subject to Microsoft's licensing terms.

## Links

*   [Original VibeVoice Repository](https://github.com/microsoft/VibeVoice) (Official Microsoft VibeVoice repository - currently unavailable).

## Credits

*   **VibeVoice Model:** Microsoft Research
*   **ComfyUI Integration:** Fabio Sarracino
*   **Base Model:** Built on Qwen2.5 architecture

## Support and Contributing

*   For issues, check the Troubleshooting section and ComfyUI logs. Open an issue with detailed information.
*   Contributions are welcome!  Follow the guidelines outlined in the repository.

## Changelog

*(The changelog from the original README has been retained here)*