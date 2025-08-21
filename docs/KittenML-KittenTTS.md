# Kitten TTS: Lightweight & High-Quality Text-to-Speech

**Bring realistic, high-quality voice synthesis to your projects with Kitten TTS, a compact and efficient open-source model.**

[Check out the original repository on GitHub](https://github.com/KittenML/KittenTTS)

## Key Features of Kitten TTS

*   **Ultra-Lightweight:**  Model size is under 25MB, perfect for resource-constrained environments.
*   **CPU-Optimized:**  Runs flawlessly on any device without requiring a GPU.
*   **High-Quality Voices:**  Choose from several premium voice options for a natural and engaging listening experience.
*   **Fast Inference:** Optimized for real-time speech synthesis, providing quick results.

## Getting Started with Kitten TTS

### Installation

Install Kitten TTS easily using pip:

```bash
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
```

### Basic Usage

Here's a quick example to get you started:

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

Kitten TTS is designed to work on literally any device.

##  Future Development

*   [x] Release a preview model
*   [ ] Release the fully trained model weights
*   [ ] Release mobile SDK
*   [ ] Release web version

## Get Support

*   [Join our Discord](https://discord.com/invite/VJ86W4SURW)
*   [For custom support - fill this form ](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
*   Email the creators with any questions : info@stellonlabs.com