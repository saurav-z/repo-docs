# VibeVoice ComfyUI Nodes: Unleash Realistic Text-to-Speech in Your Workflows

**Transform your ComfyUI workflows with the power of Microsoft's VibeVoice!** This comprehensive integration lets you generate high-quality, natural-sounding speech, including single-speaker voices, dynamic multi-speaker conversations, and voice cloning, all within ComfyUI.  [View the original repository](https://github.com/Enemyx-net/VibeVoice-ComfyUI).

**Key Features:**

*   🎤 **High-Fidelity Speech Synthesis:** Generate lifelike speech from text using advanced VibeVoice models.
*   🗣️ **Multi-Speaker Conversations:** Create engaging dialogues with up to four distinct voices.
*   🔊 **Voice Cloning:** Replicate voices from audio samples for personalized results.
*   📄 **Text File Loading:** Import scripts directly from text files for efficient workflow management.
*   🔄 **Node Chaining:** Seamlessly integrate VibeVoice nodes with other ComfyUI elements for complex audio production.
*   ⚙️ **Flexible Configuration:** Fine-tune your output with control over temperature, sampling, and guidance scale.
*   🚀 **Multiple Model Options:** Choose the best model for your needs, including VibeVoice-1.5B (fast, low memory), VibeVoice-Large (best quality, ~17GB VRAM), and VibeVoice-Large-Quant-4Bit (balanced, ~7GB VRAM).
*   ⚡ **Optimized Performance:** Benefit from attention mechanisms, adjustable diffusion steps, and VRAM management options.
*   🍎 **Apple Silicon Support:** Experience native GPU acceleration on M1/M2/M3 Macs via MPS.
*   💾 **4-Bit Quantization:** Reduces memory usage with minimal quality loss.

## Getting Started

### Installation (Automatic - Recommended)

1.  **Navigate to your ComfyUI custom nodes folder:**
    ```bash
    cd ComfyUI/custom_nodes
    ```

2.  **Clone the repository:**
    ```bash
    git clone https://github.com/Enemyx-net/VibeVoice-ComfyUI
    ```

3.  **Restart ComfyUI**: The necessary dependencies will automatically install on the first use of the nodes.

## Core Functionality: Available Nodes

### 1. VibeVoice Load Text From File
*   **Description:** Loads text content from `.txt` files within ComfyUI's input, output, or temp directories.
*   **Output:** Delivers text string to TTS nodes.

### 2. VibeVoice Single Speaker
*   **Description:** Generates speech from text using a single voice. Supports voice cloning.
*   **Inputs:**
    *   `text`: Input text (direct or from Load Text node).
    *   `model`: Choose from VibeVoice-1.5B, VibeVoice-Large, or VibeVoice-Large-Quant-4Bit.
    *   (Optional) `voice_to_clone`: Audio input for voice cloning.
*   **Parameters:** `attention_type`, `free_memory_after_generate`, `diffusion_steps`, `seed`, `cfg_scale`, `use_sampling`, `temperature`, `top_p`.

### 3. VibeVoice Multiple Speakers
*   **Description:** Generates multi-speaker conversations, ideal for dialogues and storytelling.
*   **Inputs:**
    *   `text`: Text input with speaker labels (e.g., `[1]: Hello...`, `[2]: Hi...`).
    *   `model`: Select from VibeVoice-1.5B, VibeVoice-Large, or VibeVoice-Large-Quant-4Bit.
    *   (Optional) `speaker1_voice` to `speaker4_voice`: Audio inputs for voice cloning per speaker.
*   **Parameters:** Same as Single Speaker: `attention_type`, `free_memory_after_generate`, `diffusion_steps`, `seed`, `cfg_scale`, `use_sampling`, `temperature`, `top_p`.

### 4. VibeVoice Free Memory
*   **Description:** Manually releases loaded VibeVoice models from memory. Use for VRAM/RAM management in complex workflows.
*   **Input:** `audio` (connect to trigger memory cleanup).
*   **Output:** `audio` (passes input audio through).

## Text Formatting for Multi-Speaker

*   **Format your text using `[N]:` notation**, where `N` represents the speaker number (1-4).
    ```
    [1]: Hello, how are you today?
    [2]: I'm doing great, thanks!
    ```
*   The system automatically detects the number of speakers.
*   Optional voice samples can be used for each speaker for voice cloning.

## Model Details

### VibeVoice-1.5B
*   **Size:** ~5GB
*   **Speed:** Fast inference
*   **Quality:** Good for single speakers
*   **Use Case:** Quick prototyping, individual voices.

### VibeVoice-Large
*   **Size:** ~17GB
*   **Speed:** Optimized, slower inference
*   **Quality:** Best available quality
*   **Use Case:** Highest quality production, multi-speaker conversations.

### VibeVoice-Large-Quant-4Bit
*   **Size:** ~7GB
*   **Speed:** Balanced inference
*   **Quality:** Good quality
*   **Use Case:** Good quality production with less VRAM, multi-speaker conversations
*   **Note**: Quantized by DevParker

*Models are automatically downloaded on first use and cached in `ComfyUI/models/vibevoice/`.*

## Generation Modes

*   **Deterministic Mode:** `use_sampling = False` (Default). Produces consistent and stable output, ideal for production.
*   **Sampling Mode:** `use_sampling = True`. Introduces more variation, utilizing `temperature` and `top_p` parameters for creative exploration.

## Voice Cloning Guide

1.  Connect an audio node to `voice_to_clone` (single speaker) or `speaker1_voice`, `speaker2_voice`, etc. (multi-speaker).
2.  The model will try to match the voice characteristics.
3.  **Voice Sample Requirements**: Clear audio with minimal noise. Minimum 3-10 seconds, recommended at least 30 seconds for higher quality. Automatically resampled to 24kHz.

## Tips for Best Results

1.  **Text Preparation:** Use punctuation, break up long texts. For multi-speaker, use clear speaker transitions.
2.  **Model Selection:**  1.5B for speed, Large for best quality and multi-speaker.
3.  **Seed Management:** Save seeds for consistent character voices. Use random seeds if the default doesn't work.
4.  **Performance:** GPU recommended for faster inference. First run downloads the models, subsequent runs use cached models.

## System Requirements

*   **Hardware:**
    *   **Minimum:** 8GB VRAM (for VibeVoice-1.5B)
    *   **Recommended:** 17GB+ VRAM (for VibeVoice-Large)
    *   **RAM:** 16GB+ system memory
*   **Software:**
    *   Python 3.8+
    *   PyTorch 2.0+
    *   CUDA 11.8+ (for GPU acceleration)
    *   Transformers 4.51.3+
    *   ComfyUI (latest version)

## Troubleshooting

*   **Installation:** Ensure ComfyUI's environment is used.  Try manual installation if automatic fails. Restart ComfyUI.
*   **Generation:** Use deterministic mode for stable voices. Check text formatting for multi-speaker. Check speaker numbers.
*   **Memory:** Large models require significant VRAM. Use the 1.5B model for lower VRAM systems. Models use bfloat16 precision.

## Examples

*   **Single Speaker:**
    ```
    Text: "Welcome to our presentation..."
    Model: VibeVoice-1.5B
    cfg_scale: 1.3
    use_sampling: False
    ```

*   **Two Speakers:**
    ```
    [1]: Have you seen the new AI developments?
    [2]: Yes, they're quite impressive!
    ```

*   **Four Speaker Conversation:**
    ```
    [1]: Welcome everyone...
    [2]: Thanks for having us!
    [3]: Glad to be here.
    [4]: Looking forward to the discussion.
    ```

## Performance Benchmarks
| Model                  | VRAM Usage | Context Length | Max Audio Duration |
|------------------------|------------|----------------|-------------------|
| VibeVoice-1.5B         | ~8GB | 64K tokens | ~90 minutes |
| VibeVoice-Large | ~17GB | 32K tokens | ~45 minutes |
| VibeVoice-Large-Quant-4Bit | ~7GB | 32K tokens | ~45 minutes |

## Known Limitations

*   Maximum 4 speakers in multi-speaker mode.
*   Best results with English and Chinese text.
*   Some seeds may produce unstable output.
*   Background music generation cannot be directly controlled.

## License

The ComfyUI wrapper is released under the MIT License. See the LICENSE file for details. *Note: The VibeVoice model is subject to Microsoft's licensing terms.  Check the original VibeVoice repository for full model license details.*

## Links

*   [Original VibeVoice Repository](https://github.com/microsoft/VibeVoice) - Official Microsoft VibeVoice repository (currently unavailable)

## Credits

*   **VibeVoice Model:** Microsoft Research
*   **ComfyUI Integration:** Fabio Sarracino
*   **Base Model**: Built on Qwen2.5 architecture

## Support

1.  Check the troubleshooting section.
2.  Review ComfyUI logs.
3.  Ensure VibeVoice is installed correctly.
4.  Open an issue with detailed error information.

## Contributing

Contributions are welcome! Please:
1.  Test thoroughly.
2.  Follow existing code style.
3.  Update documentation.
4.  Submit pull requests with clear descriptions.

## Changelog
(See Original README)