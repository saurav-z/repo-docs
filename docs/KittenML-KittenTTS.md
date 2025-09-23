# Kitten TTS: Realistic Text-to-Speech for Everyone 🗣️

**Kitten TTS is a lightweight, open-source text-to-speech model that delivers high-quality voice synthesis with minimal resources.** (See the [original repository](https://github.com/KittenML/KittenTTS) for more details.)

*Currently in developer preview.*

**Join the Kitten TTS Community:**

*   [Join our Discord](https://discord.com/invite/VJ86W4SURW)
*   [Request Custom Support](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
*   Email the creators with any questions: [info@stellonlabs.com](mailto:info@stellonlabs.com)

## Key Features

*   **Ultra-Lightweight:** The model is less than 25MB, making it perfect for various applications.
*   **CPU-Optimized:** Kitten TTS runs efficiently on any device without requiring a GPU.
*   **High-Quality Voices:** Experience premium voice options for a natural and engaging listening experience.
*   **Fast Inference:** Optimized for real-time speech synthesis, ensuring quick and responsive results.

## Quick Start Guide

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

Kitten TTS is designed to work on any device, making it accessible to everyone.

## Roadmap / Future Development

*   \[x] Release a preview model
*   \[ ] Release the fully trained model weights
*   \[ ] Release mobile SDK
*   \[ ] Release web version