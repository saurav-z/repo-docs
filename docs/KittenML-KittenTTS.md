# Kitten TTS: Generate Realistic Speech with a Tiny, Powerful Model

**Kitten TTS revolutionizes text-to-speech by offering high-quality, realistic voices with an incredibly lightweight and efficient model.**

[Visit the original repository on GitHub](https://github.com/KittenML/KittenTTS)

**Currently in Developer Preview**

[Join our Discord](https://discord.com/invite/VJ86W4SURW) | [Request Custom Support](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview) | [Contact Us](mailto:info@stellonlabs.com)

## Key Features

*   **Ultra-Lightweight Design:**  The model size is under 25MB, making it ideal for deployment on various devices.
*   **CPU-Optimized Performance:**  Run Kitten TTS seamlessly without a GPU, ensuring accessibility on any device.
*   **High-Quality Voices:** Experience premium voice options for a natural and engaging listening experience.
*   **Fast Inference Speed:**  Optimized for real-time speech synthesis, enabling quick and efficient audio generation.

## Getting Started

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

Kitten TTS is designed to work on a wide range of systems.

## Roadmap / Checklist

*   \[x] Release a preview model
*   \[ ] Release the fully trained model weights
*   \[ ] Release mobile SDK
*   \[ ] Release web version