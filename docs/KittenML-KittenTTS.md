# Kitten TTS: Unleash Realistic Text-to-Speech with Tiny Models 🗣️

Tired of bulky text-to-speech models? **Kitten TTS offers a lightweight, high-quality, and CPU-optimized solution for generating realistic speech from text.**

[View the original Kitten TTS repository on GitHub](https://github.com/KittenML/KittenTTS)

*Currently in developer preview*

[Join our Discord community](https://discord.com/invite/VJ86W4SURW)

[Get custom support - fill this form](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)

Email us with any questions: info@stellonlabs.com

## Key Features of Kitten TTS

*   **Ultra-Lightweight:**  Model size under 25MB, ideal for resource-constrained environments.
*   **CPU-Optimized:** Runs smoothly on any device, without the need for a GPU.
*   **High-Quality Voices:** Access a range of premium voice options for diverse applications.
*   **Fast Inference:** Experience real-time speech synthesis with optimized performance.

## Getting Started with Kitten TTS

### Installation

Install Kitten TTS using pip:

```bash
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
```

### Basic Usage

Generate speech with just a few lines of code:

```python
from kittentts import KittenTTS
m = KittenTTS("KittenML/kitten-tts-nano-0.1")

audio = m.generate("This high quality TTS model works without a GPU", voice='expr-voice-2-f' )

# available_voices : [  'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',  'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f' ]

# Save the audio
import soundfile as sf
sf.write('output.wav', audio, 24000)
```

## System Requirements

Kitten TTS is designed to work on virtually any device, making it a versatile choice for various projects.

## Future Development

*   [x] Release a preview model
*   [ ] Release the fully trained model weights
*   [ ] Release mobile SDK
*   [ ] Release web version