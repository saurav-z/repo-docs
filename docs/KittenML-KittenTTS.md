# Kitten TTS: Realistic Text-to-Speech for Everyone 🗣️

**Kitten TTS is an open-source, ultra-lightweight text-to-speech (TTS) model, enabling high-quality voice synthesis on any device.**  [Learn more on the original GitHub repository](https://github.com/KittenML/KittenTTS).

*Currently in developer preview*

[Join our Discord community](https://discord.com/invite/VJ86W4SURW) for support and updates.

[Request Custom Support](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)

For any questions or inquiries, please contact us at info@stellonlabs.com

## Key Features of Kitten TTS

*   **Ultra-Lightweight:** Model size under 25MB, perfect for resource-constrained environments.
*   **CPU-Optimized:**  Runs seamlessly on any device, without requiring a GPU.
*   **High-Quality Voices:** Experience natural-sounding speech with various premium voice options.
*   **Fast Inference:** Optimized for real-time speech generation.

## Getting Started with Kitten TTS

### Installation

Install the Kitten TTS package using pip:

```bash
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
```

### Basic Usage Example

Generate speech from text using the following Python code snippet:

```python
from kittentts import KittenTTS
m = KittenTTS("KittenML/kitten-tts-nano-0.2")

audio = m.generate("This high quality TTS model works without a GPU", voice='expr-voice-2-f' )

# available_voices : [  'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',  'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f' ]

# Save the audio
import soundfile as sf
sf.write('output.wav', audio, 24000)
```

## System Requirements

Kitten TTS is designed to work on virtually any device.

## Roadmap & Future Development

*   [x] Release a preview model
*   [ ] Release the fully trained model weights
*   [ ] Release mobile SDK
*   [ ] Release web version