# KittenTTS: Realistic Text-to-Speech for Everyone 😻

**Tired of bulky and resource-intensive text-to-speech (TTS) models?** KittenTTS offers a groundbreaking, lightweight, and high-quality TTS solution that's perfect for any project. (See the original repo [here](https://github.com/KittenML/KittenTTS).)

## Key Features

*   **Ultra-Lightweight:** Model size under 25MB for easy deployment.
*   **CPU-Optimized:** Runs smoothly on any device, no GPU required.
*   **High-Quality Voices:** Experience natural-sounding speech with various premium voice options.
*   **Fast Inference:** Optimized for real-time speech synthesis and rapid generation.

## Quick Start

Get started with KittenTTS in minutes!

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

KittenTTS is designed to work on virtually any system.

## Roadmap

*   \[x] Release a preview model
*   \[ ] Release the fully trained model weights
*   \[ ] Release a mobile SDK
*   \[ ] Release a web version

## Get Involved

*   [Join our Discord](https://discord.com/invite/VJ86W4SURW)
*   [For custom support - fill this form ](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
*   Email us with any questions: [info@stellonlabs.com](mailto:info@stellonlabs.com)