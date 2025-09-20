# KittenTTS: Realistic Text-to-Speech with Minimal Resources

**Transform text into natural-sounding speech effortlessly with KittenTTS, a lightweight and high-quality open-source text-to-speech model.**  [See the original repository](https://github.com/KittenML/KittenTTS).

## Key Features

*   **Ultra-Lightweight:** Enjoy a model size under 25MB, perfect for deployment on various devices.
*   **CPU-Optimized:** Run the model seamlessly on any device without requiring a GPU.
*   **High-Quality Voices:** Access several premium voice options for diverse speech synthesis needs.
*   **Fast Inference:** Experience optimized performance for real-time speech generation.

## Getting Started

### Installation

Install KittenTTS directly from the releases using pip:

```bash
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
```

### Basic Usage

Here's how to get started with KittenTTS:

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

KittenTTS is designed to work universally. It has minimal system requirements and runs on any device.

## Resources

*   **Join our Discord:** [Discord Link](https://discord.com/invite/VJ86W4SURW)
*   **Custom Support:** [Support Form](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
*   **Email:** info@stellonlabs.com

## Development Roadmap

*   [x] Release a preview model
*   [ ] Release the fully trained model weights
*   [ ] Release mobile SDK
*   [ ] Release web version