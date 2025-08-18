# Kitten TTS: Realistic Text-to-Speech for Everyone 😻

**Unlock high-quality, natural-sounding speech synthesis with Kitten TTS, an open-source, lightweight text-to-speech model.**

[Explore the original KittenTTS repository on GitHub](https://github.com/KittenML/KittenTTS)

Currently in Developer Preview. Join our [Discord](https://discord.com/invite/VJ86W4SURW) or contact us directly via email: info@stellonlabs.com. For custom support, please fill out this [form](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview).

## Key Features

*   **Ultra-Lightweight:** The model size is under 25MB, ideal for efficient deployment.
*   **CPU-Optimized:** Run Kitten TTS without a GPU on any device.
*   **High-Quality Voices:** Choose from a selection of premium voice options.
*   **Fast Inference:** Optimized for real-time speech synthesis, ensuring quick generation.

## Quick Start Guide

### Installation

Install the Kitten TTS package directly from GitHub releases:

```bash
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
```

### Basic Usage

Here's how to start using Kitten TTS to generate speech:

```python
from kittentts import KittenTTS
m = KittenTTS("KittenML/kitten-tts-nano-0.1")

audio = m.generate("This high quality TTS model works without a GPU", voice='expr-voice-2-f' )

# Available voices: ['expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f', 'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f']

# Save the audio
import soundfile as sf
sf.write('output.wav', audio, 24000)
```

## System Requirements

Kitten TTS is designed to be incredibly versatile and works on nearly any system.

## Development Roadmap

*   \[x] Release a preview model
*   \[ ] Release the fully trained model weights
*   \[ ] Release mobile SDK
*   \[ ] Release web version