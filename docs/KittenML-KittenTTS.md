# Kitten TTS: Unleash Realistic Text-to-Speech with Lightweight Performance

Tired of bulky TTS models? Kitten TTS offers a cutting-edge, open-source solution for high-quality, realistic voice synthesis in a compact package. Explore the original repository here: [KittenML/KittenTTS](https://github.com/KittenML/KittenTTS)

## Key Features

*   **Ultra-Lightweight:** At less than 25MB, Kitten TTS is designed for efficient deployment on any device.
*   **CPU-Optimized:** No GPU required! Runs seamlessly on any CPU, eliminating hardware dependencies.
*   **High-Quality Voices:** Access a range of premium voice options for diverse applications.
*   **Fast Inference:** Optimized for real-time speech synthesis, ensuring a smooth user experience.

## Quick Start Guide

### Installation

```bash
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
```

### Basic Usage

```python
from kittentts import KittenTTS
m = KittenTTS("KittenML/kitten-tts-nano-0.1")

audio = m.generate("This high quality TTS model works without a GPU", voice='expr-voice-2-f' )

# available_voices : [  'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',  'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f' ]

# Save the audio
import soundfile as sf
sf.write('output.wav', audio, 24000)
```

## System Requirements

Kitten TTS is designed to work virtually everywhere.

## Future Development (Checklist)

*   \[x] Release a preview model
*   \[ ] Release the fully trained model weights
*   \[ ] Release mobile SDK
*   \[ ] Release web version

## Get Involved

*   [Join our Discord](https://discord.com/invite/VJ86W4SURW)
*   [For Custom Support - fill this form](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
*   Email: info@stellonlabs.com