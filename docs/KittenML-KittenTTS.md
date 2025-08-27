# Kitten TTS: Realistic Text-to-Speech with Lightweight Design

**Transform text into natural-sounding speech with Kitten TTS, an open-source, ultra-lightweight text-to-speech model perfect for any device.**

[Visit the original repository on GitHub](https://github.com/KittenML/KittenTTS)

*Currently in developer preview*

[Join our Discord Community](https://discord.com/invite/VJ86W4SURW)

[Get Custom Support](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)

For any questions, reach out to the creators: [info@stellonlabs.com](mailto:info@stellonlabs.com)

## Key Features of Kitten TTS

*   **Ultra-Lightweight:** The model size is less than 25MB, making it ideal for deployment on various devices.
*   **CPU-Optimized:** Run Kitten TTS seamlessly without a GPU, on any device.
*   **High-Quality Voices:** Experience premium voice options for diverse applications.
*   **Fast Inference:** Benefit from optimized performance for real-time speech synthesis.

## Getting Started with Kitten TTS

### Installation

Install Kitten TTS with a simple pip command:

```bash
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
```

### Basic Usage

Here's a quick example to get you started:

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

Kitten TTS is designed to run on virtually any device.

## Roadmap & Future Development

*   [x] Release a preview model
*   [ ] Release the fully trained model weights
*   [ ] Release mobile SDK
*   [ ] Release web version