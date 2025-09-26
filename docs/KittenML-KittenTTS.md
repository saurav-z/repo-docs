# KittenTTS: Unleash Realistic Text-to-Speech with Lightweight Efficiency

KittenTTS offers a groundbreaking, open-source text-to-speech (TTS) solution, delivering high-quality voice synthesis in a remarkably compact package. [View the original repository on GitHub](https://github.com/KittenML/KittenTTS).

## Key Features

*   **Ultra-Lightweight Design:** The model is under 25MB, ideal for deployment on various devices.
*   **CPU-Optimized Performance:** KittenTTS operates efficiently on CPUs, eliminating the need for a GPU.
*   **High-Quality Voices:** Experience premium voice options for natural-sounding speech.
*   **Fast Inference:** Optimized for real-time speech synthesis, providing a seamless user experience.

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

KittenTTS is designed to be highly accessible and runs smoothly on virtually any system.

## Roadmap

*   [x] Release a preview model
*   [ ] Release the fully trained model weights
*   [ ] Release mobile SDK
*   [ ] Release web version

## Join the Community

*   [Join our Discord](https://discord.com/invite/VJ86W4SURW)
*   [For custom support - fill this form](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
*   Email the creators with any questions: info@stellonlabs.com