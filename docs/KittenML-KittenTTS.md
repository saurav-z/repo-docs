# KittenTTS: Lightweight, High-Quality Text-to-Speech (TTS)

**Experience realistic voice synthesis on any device with KittenTTS, the open-source, ultra-lightweight text-to-speech model!** This innovative model delivers exceptional audio quality while remaining incredibly efficient.  [View the original repository on GitHub](https://github.com/KittenML/KittenTTS).

## Key Features:

*   **Ultra-Lightweight:**  The model size is less than 25MB, making it ideal for resource-constrained environments.
*   **CPU-Optimized:**  Runs seamlessly on any device without a GPU, expanding accessibility.
*   **High-Quality Voices:**  Enjoy several premium voice options for diverse and engaging audio output.
*   **Fast Inference:** Optimized for real-time speech synthesis, providing a responsive user experience.

## Getting Started:

### Installation:

```bash
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
```

### Basic Usage:

```python
from kittentts import KittenTTS
m = KittenTTS("KittenML/kitten-tts-nano-0.2")

audio = m.generate("This high quality TTS model works without a GPU", voice='expr-voice-2-f' )

# available_voices : [  'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',  'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f' ]

# Save the audio
import soundfile as sf
sf.write('output.wav', audio, 24000)
```

## System Requirements:

KittenTTS is designed to work on virtually any device.

## Roadmap / Future Development:

*   [x] Release a preview model
*   [ ] Release the fully trained model weights
*   [ ] Release mobile SDK
*   [ ] Release web version

## Join the Community:

*   [Join our Discord](https://discord.com/invite/VJ86W4SURW)
*   [For custom support - fill this form](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
*   Email the creators with any questions: info@stellonlabs.com