# VibeVoice ComfyUI Integration: Unleash Realistic Text-to-Speech in Your Workflows

**Effortlessly generate natural-sounding voices and multi-speaker conversations directly within ComfyUI using Microsoft's VibeVoice models. Check out the [original repo](https://github.com/Enemyx-net/VibeVoice-ComfyUI) for more information and updates.**

## Key Features

*   🎤 **Single Speaker TTS:** Generate speech from text with optional voice cloning for a personalized touch.
*   👥 **Multi-Speaker Conversations:** Create engaging dialogues with up to 4 distinct voices.
*   🗣️ **Voice Cloning:** Clone voices from audio samples to replicate unique vocal characteristics.
*   📁 **Text File Loading:** Easily load and process scripts from text files.
*   🚀 **Model Options:** Choose between two model sizes for optimal speed and quality (1.5B and Large).
*   ⚙️ **Flexible Configuration:** Fine-tune generation with customizable parameters like temperature, sampling, and guidance scale.
*   ⚡ **Performance Optimization:** Leverage attention mechanisms, diffusion steps, and memory management for efficient processing.

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

3.  Restart ComfyUI. The necessary dependencies will be automatically installed upon first use of the nodes.

## Available Nodes

### 1. VibeVoice Load Text From File

*   **Description:** Loads text content from .txt files.
*   **Input:** File path (from ComfyUI's input/output/temp directories).
*   **Output:** Text string.

### 2. VibeVoice Single Speaker

*   **Description:** Generates speech using a single voice.
*   **Inputs:**
    *   `text`: Input text (direct or from "Load Text" node).
    *   `model`: VibeVoice-1.5B or VibeVoice-Large.
    *   `voice_to_clone` (Optional): Audio input for voice cloning.
    *   Other parameters for fine-tuning (seed, cfg_scale, temperature, etc.)
*   **Outputs:** Audio

### 3. VibeVoice Multiple Speakers

*   **Description:** Generates multi-speaker conversations.
*   **Speaker Format:** Use `[N]:` notation (N = 1-4) in your text.
*   **Inputs:**
    *   `text`: Input text with speaker labels.
    *   `model`: VibeVoice-1.5B or VibeVoice-Large.
    *   `speaker1_voice` to `speaker4_voice` (Optional): Audio inputs for voice cloning.
    *   Other parameters for fine-tuning (seed, cfg_scale, temperature, etc.)
*   **Outputs:** Audio

### 4. VibeVoice Free Memory

*   **Description:** Frees loaded VibeVoice models from memory. Useful for managing VRAM in complex workflows.
*   **Input:** `audio` (Connect audio output to trigger cleanup).
*   **Output:** `audio` (passes through the input audio).

## Multi-Speaker Text Formatting

Format your text for multi-speaker generation using the following notation:

```
[1]: Hello, how are you today?
[2]: I'm doing great, thanks for asking!
```

*   Use `[1]:`, `[2]:`, `[3]:`, and `[4]:` to label speakers (max 4).
*   The system automatically detects the number of speakers.
*   Optional voice samples for each speaker can be used for cloning.

## Model Information

### VibeVoice-1.5B

*   **Size:** ~5GB download
*   **Speed:** Faster inference.
*   **Quality:** Good for single-speaker tasks.
*   **Use Case:** Quick prototyping and single-voice applications.

### VibeVoice-Large

*   **Size:** ~17GB download
*   **Speed:** Optimized for performance.
*   **Quality:** Highest quality audio.
*   **Use Case:** Production-quality output and multi-speaker conversations.

*Models are downloaded on first use to the `ComfyUI/models/vibevoice/` directory.*

## Generation Modes

### Deterministic Mode (Default)

*   `use_sampling = False`
*   Produces consistent and stable results.
*   Recommended for production environments.

### Sampling Mode

*   `use_sampling = True`
*   Offers more variation in output.
*   Utilizes temperature and top\_p parameters.
*   Ideal for creative exploration.

## Voice Cloning

To clone a voice, connect an audio node:

1.  To the `voice_to_clone` input (Single Speaker node).
2.  Or, to the `speaker1_voice`, `speaker2_voice`, etc. inputs (Multi-Speaker node).

**Voice Sample Requirements:**

*   Clear audio, minimal background noise.
*   Minimum 3-10 seconds, 30 seconds recommended.
*   Automatically resampled to 24kHz.

## Tips for Best Results

1.  **Text Preparation:** Use correct punctuation, break long texts into paragraphs, and clearly define speaker transitions in multi-speaker mode.
2.  **Model Selection:** Choose 1.5B for speed and single-speaker tasks, and Large for high-quality results.
3.  **Seed Management:** Save effective seeds for consistent voice characteristics and try random seeds for variability.
4.  **Performance:** Remember that models are downloaded the first time. Use a GPU for faster processing.

## System Requirements

*   **Hardware:**
    *   Minimum: 8GB VRAM (VibeVoice-1.5B).
    *   Recommended: 17GB+ VRAM (VibeVoice-Large).
    *   RAM: 16GB+ system memory.
*   **Software:**
    *   Python 3.8+.
    *   PyTorch 2.0+.
    *   CUDA 11.8+ (for GPU acceleration).
    *   Transformers 4.51.3+.
    *   ComfyUI (latest version).

## Troubleshooting

*   **Installation Issues:** Ensure you are in the ComfyUI's Python environment, try manual installation, and restart ComfyUI.
*   **Generation Issues:** Test deterministic mode, check text formatting in multi-speaker mode, and ensure speaker numbers are sequential (1, 2, 3).
*   **Memory Issues:** Use the correct model for the available VRAM and utilize the Free Memory Node.

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

### Four Speaker Conversation

```
[1]: Welcome everyone to our meeting.
[2]: Thanks for having us!
[3]: Glad to be here.
[4]: Looking forward to the discussion.
[1]: Let's begin with the agenda.
```

## Performance Benchmarks

| Model             | VRAM Usage | Context Length | Max Audio Duration |
|-------------------|------------|----------------|--------------------|
| VibeVoice-1.5B    | ~8GB        | 64K tokens      | ~90 minutes       |
| VibeVoice-Large | ~16GB       | 32K tokens      | ~45 minutes       |

## Known Limitations

*   Maximum 4 speakers in multi-speaker mode.
*   Best performance with English and Chinese text.
*   Unstable output may occur with certain seeds.
*   Background music generation is not directly controlled.

## License

This ComfyUI wrapper is released under the MIT License.  See the LICENSE file for more details.

**Important Note:**  The VibeVoice model itself is subject to Microsoft's licensing terms.  VibeVoice is for research purposes only.  Check the [Microsoft VibeVoice repository](https://github.com/microsoft/VibeVoice) for full details.

## Credits

*   **VibeVoice Model:** Microsoft Research
*   **ComfyUI Integration:** Fabio Sarracino
*   **Base Model:** Built on the Qwen2.5 architecture

## Support

For issues or questions, please:

1.  Consult the troubleshooting section.
2.  Review ComfyUI logs for error messages.
3.  Confirm VibeVoice is installed correctly.
4.  Open an issue with detailed error information.

## Contributing

Contributions are welcome!  Please:

1.  Test changes thoroughly.
2.  Follow the existing code style.
3.  Update documentation as needed.
4.  Submit pull requests with clear descriptions.

## Changelog (Summarized)

*   **v1.2.1:** Bug fixes.
*   **v1.2.0:** Apple Silicon (MPS) support for GPU acceleration on Macs.
*   **v1.1.1:** Universal Transformers Compatibility.
*   **v1.1.0:** Updated model download URL, removed deprecated model.
*   **v1.0.9:** Embedded VibeVoice code for easier use.
*   **v1.0.8:** BFloat16 Compatibility Fix.
*   **v1.0.7:** Added interruption handler and bug fixes.
*   **v1.0.6:** Bug fix for audio flow.
*   **v1.0.5:** Added support for VibeVoice-Large (stable release).
*   **v1.0.4:** Improved tokenizer handling.
*   **v1.0.3:** Added `attention_type` and `diffusion_steps` parameters.
*   **v1.0.2:** Added `free_memory_after_generate` toggle and dedicated "Free Memory Node".
*   **v1.0.1:** Fixed line breaks in speaker text.
*   **v1.0.0:** Initial release.