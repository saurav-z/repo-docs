# Kitten TTS: Realistic Text-to-Speech for Everyone

**Experience high-quality, realistic text-to-speech with Kitten TTS, a lightweight and CPU-optimized model perfect for any device!**  ([Original Repository](https://github.com/KittenML/KittenTTS))

## Key Features of Kitten TTS:

*   **Ultra-Lightweight Design:**  The model size is less than 25MB, making it perfect for deployment on various devices.
*   **CPU-Optimized Performance:**  Kitten TTS runs efficiently without a GPU, ensuring broad compatibility.
*   **High-Quality Voice Options:** Access several premium voice options for diverse and engaging audio output.
*   **Fast Inference Speed:** Optimized for real-time speech synthesis, providing a seamless user experience.

## Getting Started with Kitten TTS

### Installation

```bash
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
```

### Basic Usage Example

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

Kitten TTS is designed to work on virtually any system.

## Support and Community

*   **Join our Discord:** [Discord Link](https://discord.com/invite/VJ86W4SURW)
*   **Custom Support:**  [Support Form](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
*   **Email:**  info@stellonlabs.com

## Development Roadmap

*   \[x] Release a preview model
*   \[ ] Release the fully trained model weights
*   \[ ] Release mobile SDK
*   \[ ] Release web version