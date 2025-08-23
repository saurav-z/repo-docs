# Kitten TTS: Unleash Realistic Text-to-Speech on Any Device (😻)

**Kitten TTS** is an open-source text-to-speech (TTS) model that brings high-quality voice synthesis to your fingertips, all while being incredibly lightweight and efficient. [View the original repository on GitHub](https://github.com/KittenML/KittenTTS).

**Currently in developer preview.**

[Join our Discord community](https://discord.com/invite/VJ86W4SURW)
[Request custom support](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
For any questions, email us at: info@stellonlabs.com

## Key Features:

*   **Ultra-Lightweight:** The model size is under 25MB, making it perfect for deployment on various devices.
*   **CPU-Optimized:** Run Kitten TTS without a GPU, ensuring compatibility with nearly any device.
*   **High-Quality Voices:** Access several premium voice options for a natural-sounding experience.
*   **Fast Inference:** Optimized for real-time speech synthesis, providing quick and responsive audio generation.

## Quick Start:

### Installation

```bash
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
```

### Basic Usage

```python
from kittentts import KittenTTS
m = KittenTTS("KittenML/kitten-tts-nano-0.1")

audio = m.generate("This high quality TTS model works without a GPU", voice='expr-voice-2-f' )

# available_voices : [  'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',  'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f' ]

# Save the audio
import soundfile as sf
sf.write('output.wav', audio, 24000)
```

## System Requirements:

Kitten TTS is designed to work on virtually any device, with no specific hardware requirements.

## Development Roadmap:

*   \[x] Release a preview model
*   \[ ] Release the fully trained model weights
*   \[ ] Release mobile SDK
*   \[ ] Release web version