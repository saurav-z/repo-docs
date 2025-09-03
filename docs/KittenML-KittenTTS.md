# Kitten TTS: Realistic Text-to-Speech for Everyone

**Create stunning, high-quality speech from text with Kitten TTS, an open-source, lightweight text-to-speech model.**

[Visit the original repository on GitHub](https://github.com/KittenML/KittenTTS)

Kitten TTS offers a powerful and accessible solution for realistic voice synthesis, perfect for a wide range of applications.

## Key Features of Kitten TTS:

*   **Ultra-Lightweight:**  The model is less than 25MB, making it easy to deploy on various devices.
*   **CPU-Optimized:** No GPU is required; it runs smoothly on any device, even those with limited resources.
*   **High-Quality Voices:** Choose from several premium voice options to suit your needs.
*   **Fast Inference:** Optimized for real-time speech synthesis, providing a seamless user experience.

## Getting Started with Kitten TTS:

### Installation

Install Kitten TTS with a simple pip command:

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

Kitten TTS is designed to work virtually everywhere, requiring minimal system resources.

## Roadmap

*   [x] Release a preview model
*   [ ] Release the fully trained model weights
*   [ ] Release mobile SDK
*   [ ] Release web version

## Get Support

*   [Join our Discord](https://discord.com/invite/VJ86W4SURW)
*   [Request Custom Support](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
*   Email the creators: info@stellonlabs.com