# KittenTTS: Realistic Text-to-Speech with Lightweight Deployment

**Create captivating audio experiences with KittenTTS, an open-source, high-quality text-to-speech model that's both lightweight and easy to deploy.**  ([View the Original Repo](https://github.com/KittenML/KittenTTS))

**Status:** Developer Preview

**Join the Community:** [Discord](https://discord.com/invite/VJ86W4SURW)

**Contact Us:** [Support Form](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview) | [Email](mailto:info@stellonlabs.com)

## Key Features

*   **Ultra-Lightweight:** Model size under 25MB for efficient deployment.
*   **CPU-Optimized:** Runs seamlessly on any device without requiring a GPU.
*   **High-Quality Voices:** Enjoy a selection of premium voice options for diverse applications.
*   **Fast Inference:** Optimized for real-time speech synthesis, ensuring a responsive user experience.

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

KittenTTS is designed to work on virtually any system.

## Development Roadmap

*   \[x] Release a preview model
*   \[ ] Release the fully trained model weights
*   \[ ] Release mobile SDK
*   \[ ] Release web version