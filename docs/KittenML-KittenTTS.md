# Kitten TTS: Lightweight, High-Quality Text-to-Speech 😻

**Transform text into natural-sounding speech with Kitten TTS, an open-source, CPU-optimized text-to-speech model that delivers impressive results with minimal resources.**

[View the original repository on GitHub](https://github.com/KittenML/KittenTTS)

*Currently in developer preview*

[Join our Discord community](https://discord.com/invite/VJ86W4SURW)

[Request custom support](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)

For any questions, contact the creators: info@stellonlabs.com

## Key Features:

*   **Ultra-Lightweight:** Model size under 25MB for easy deployment.
*   **CPU-Optimized:** Runs seamlessly on any device without a GPU.
*   **High-Quality Voices:** Access multiple premium voice options for diverse speech synthesis.
*   **Fast Inference:** Optimized for rapid, real-time speech generation.

## Quick Start Guide

### Installation

Install Kitten TTS using pip:

```bash
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
```

### Basic Usage

Here's how to generate speech from text:

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

Kitten TTS is designed to be incredibly versatile and works on virtually any device.

## Development Roadmap

*   \[x] Release a preview model
*   \[ ] Release the fully trained model weights
*   \[ ] Release mobile SDK
*   \[ ] Release web version