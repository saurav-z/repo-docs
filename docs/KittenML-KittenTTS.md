# Kitten TTS: Realistic Text-to-Speech for Everyone 😻

**Experience high-quality, realistic text-to-speech generation with Kitten TTS, a lightweight and open-source model perfect for any device.**

[View the original repository on GitHub](https://github.com/KittenML/KittenTTS)

[Join our Discord community](https://discord.com/invite/VJ86W4SURW)

[Get custom support](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)

For any inquiries, please contact us at info@stellonlabs.com.

## Key Features

*   **Ultra-Lightweight:** Model size under 25MB, ideal for resource-constrained environments.
*   **CPU-Optimized:**  Runs seamlessly on any device without a GPU, perfect for broad accessibility.
*   **High-Quality Voices:**  Choose from several premium voice options for engaging audio output.
*   **Fast Inference:** Optimized for real-time speech synthesis, providing a responsive user experience.

## Quick Start Guide

### Installation

Easily install Kitten TTS using pip:

```bash
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
```

### Basic Usage

Get started generating speech with just a few lines of code:

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

Kitten TTS is designed to work "literally everywhere," requiring minimal system resources.

## Roadmap

*   [x] Release a preview model
*   [ ] Release the fully trained model weights
*   [ ] Release mobile SDK
*   [ ] Release web version