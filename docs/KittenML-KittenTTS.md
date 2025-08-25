# KittenTTS: The Lightweight & Realistic Text-to-Speech Model (Open Source)

**Transform text into natural-sounding speech with KittenTTS, an open-source, ultra-lightweight text-to-speech model perfect for deployment on any device.**  [See the original repository](https://github.com/KittenML/KittenTTS)

## Key Features:

*   **Ultra-Lightweight:** At under 25MB, KittenTTS is incredibly compact.
*   **CPU-Optimized:** Runs efficiently on CPUs, eliminating the need for a GPU.
*   **High-Quality Voices:** Choose from several premium voice options.
*   **Fast Inference:** Experience real-time speech synthesis with optimized performance.

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

# Available voices: ['expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f', 'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f']

# Save the audio
import soundfile as sf
sf.write('output.wav', audio, 24000)
```

## System Requirements

KittenTTS is designed to run on virtually any device.

## Get Support and Stay Updated:

*   **Join our Discord:** [https://discord.com/invite/VJ86W4SURW](https://discord.com/invite/VJ86W4SURW)
*   **Custom Support:**  [Fill out this form](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
*   **Email:**  info@stellonlabs.com

## Development Roadmap

*   [x] Release a preview model
*   [ ] Release the fully trained model weights
*   [ ] Release mobile SDK
*   [ ] Release web version