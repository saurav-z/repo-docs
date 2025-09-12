# Kitten TTS: Realistic Text-to-Speech (TTS) for Everyone

**Bring your text to life with Kitten TTS, a lightweight, open-source text-to-speech model that delivers high-quality voice synthesis with minimal resources.** (See the original repository [here](https://github.com/KittenML/KittenTTS).)

## Key Features

*   **Ultra-Lightweight:** Model size is under 25MB, making it ideal for deployment on various devices.
*   **CPU-Optimized:** Runs efficiently without a GPU, enabling accessibility on a wide range of hardware.
*   **High-Quality Voices:** Offers several premium voice options for diverse and engaging audio output.
*   **Fast Inference:** Optimized for real-time speech synthesis, providing a seamless user experience.

## Getting Started: Quick Installation & Usage

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

Kitten TTS is designed to be highly compatible and works on virtually any system.

## Future Development Roadmap

*   [x] Release a preview model
*   [ ] Release the fully trained model weights
*   [ ] Release mobile SDK
*   [ ] Release web version

## Support & Community

*   Join our Discord community: [Discord Link](https://discord.com/invite/VJ86W4SURW)
*   For custom support, please fill out this form: [Support Form](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
*   Contact the creators: info@stellonlabs.com