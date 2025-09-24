# Kitten TTS: Unleash Realistic Text-to-Speech on Any Device 😻

**Kitten TTS** is an open-source, ultra-lightweight text-to-speech model that delivers high-quality voice synthesis with just 15 million parameters, perfect for lightweight deployment.

[View the Kitten TTS repository on GitHub](https://github.com/KittenML/KittenTTS)

*Currently in developer preview*

[Join our Discord community](https://discord.com/invite/VJ86W4SURW)
[Get custom support](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
Email us with questions: info@stellonlabs.com

## Key Features of Kitten TTS

*   **Ultra-Lightweight:** Model size under 25MB, ideal for resource-constrained environments.
*   **CPU-Optimized:** Operates efficiently without a GPU, enabling deployment on any device.
*   **High-Quality Voices:** Enjoy a selection of premium voice options for diverse use cases.
*   **Fast Inference:** Optimized for real-time speech synthesis, delivering rapid audio generation.

## Getting Started with Kitten TTS

### Installation

```bash
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
```

### Basic Usage

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

Kitten TTS is designed to run on a wide range of systems.

## Future Development

*   \[x] Release a preview model
*   \[ ] Release the fully trained model weights
*   \[ ] Release mobile SDK
*   \[ ] Release web version