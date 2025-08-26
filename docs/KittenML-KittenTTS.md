# Kitten TTS: Unleash Realistic Text-to-Speech on Any Device 🗣️

Kitten TTS is a groundbreaking, open-source text-to-speech model that delivers high-quality, realistic voice synthesis in a remarkably lightweight package. Check out the original repo for more details: [KittenML/KittenTTS](https://github.com/KittenML/KittenTTS)

*Currently in developer preview*

[Join our Discord](https://discord.com/invite/VJ86W4SURW)

[For custom support - fill this form](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)

Email the creators with any questions: info@stellonlabs.com

## Key Features of Kitten TTS

*   **Ultra-Lightweight:**  Our model size is under 25MB, perfect for resource-constrained environments.
*   **CPU-Optimized:**  Run Kitten TTS on any device without needing a GPU.
*   **High-Quality Voices:** Enjoy access to several premium voice options.
*   **Fast Inference:** Get real-time speech synthesis thanks to our optimization efforts.

## Getting Started: Quick Start Guide

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

Kitten TTS is designed to run seamlessly on virtually any system.

## Development Roadmap

*   [x] Release a preview model
*   [ ] Release the fully trained model weights
*   [ ] Release mobile SDK
*   [ ] Release web version