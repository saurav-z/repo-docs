# KittenTTS: Lightweight, High-Quality Text-to-Speech for Everyone

**KittenTTS offers a revolutionary, open-source text-to-speech model, delivering natural-sounding voices with an incredibly small footprint.** (See the original repo here: [https://github.com/KittenML/KittenTTS](https://github.com/KittenML/KittenTTS))

## Key Features of KittenTTS

*   **Ultra-Lightweight Design:** With a model size of less than 25MB, KittenTTS is perfect for resource-constrained environments.
*   **CPU-Optimized Performance:** Run the model efficiently on any device without the need for a GPU.
*   **High-Quality Voice Options:** Enjoy a selection of premium voice options for diverse and engaging audio output.
*   **Fast Inference Speed:** Optimized for real-time speech synthesis, making it ideal for interactive applications.

## Getting Started with KittenTTS

### Installation

```bash
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
```

### Basic Usage

```python
from kittentts import KittenTTS
m = KittenTTS("KittenML/kitten-tts-nano-0.2")

audio = m.generate("This high quality TTS model works without a GPU", voice='expr-voice-2-f' )

# Available Voices: ['expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',  'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f']

# Save the audio
import soundfile as sf
sf.write('output.wav', audio, 24000)
```

## System Requirements

KittenTTS is designed to run on virtually any system.

## Future Development

*   [x] Release a preview model
*   [ ] Release the fully trained model weights
*   [ ] Release mobile SDK
*   [ ] Release web version

## Stay Connected

*   [Join our Discord](https://discord.com/invite/VJ86W4SURW)
*   [Request Custom Support](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
*   Email the creators: info@stellonlabs.com