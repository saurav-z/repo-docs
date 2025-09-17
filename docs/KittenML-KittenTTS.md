# KittenTTS: Realistic Text-to-Speech with Lightweight Deployment 😻

**Experience high-quality, realistic text-to-speech (TTS) synthesis with KittenTTS, an open-source model perfect for deployment on any device.**  [Visit the original repository on GitHub](https://github.com/KittenML/KittenTTS)

## Key Features:

*   **Ultra-Lightweight:** The model size is less than 25MB, ideal for resource-constrained environments.
*   **CPU-Optimized:** KittenTTS runs efficiently without a GPU, making it accessible on a wide range of devices.
*   **High-Quality Voices:** Access several premium voice options for diverse and engaging audio output.
*   **Fast Inference:** Optimized for real-time speech synthesis, ensuring a seamless user experience.

## Getting Started Quickly

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

KittenTTS is designed to work literally everywhere.

## Stay Updated

*   [Join our Discord](https://discord.com/invite/VJ86W4SURW)
*   [For custom support - fill this form](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
*   Email us with questions: info@stellonlabs.com