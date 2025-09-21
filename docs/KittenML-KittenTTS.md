# Kitten TTS: Unleash Realistic Text-to-Speech Anywhere 😻

Kitten TTS is an open-source text-to-speech (TTS) model offering high-quality, realistic voices in a lightweight package, perfect for deployment on any device. [See the original repository](https://github.com/KittenML/KittenTTS).

*Currently in developer preview*

*Join our Discord community: [Discord Link](https://discord.com/invite/VJ86W4SURW)*

*For custom support - fill out this form: [Support Form](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)*

*Email the creators with any questions: info@stellonlabs.com*

## Key Features of Kitten TTS

*   **Ultra-Lightweight Design:** With a model size under 25MB, Kitten TTS is perfect for resource-constrained environments.
*   **CPU-Optimized:** Runs seamlessly on any device without the need for a GPU.
*   **High-Quality Voices:** Experience a selection of premium voice options.
*   **Fast Inference:** Optimized for real-time speech synthesis and rapid audio generation.

## Getting Started with Kitten TTS

### Installation

Install Kitten TTS using pip:

```bash
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
```

### Basic Usage Example

Generate speech from text with the following Python code snippet:

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

## Future Development Roadmap

*   [x] Release a preview model
*   [ ] Release the fully trained model weights
*   [ ] Release mobile SDK
*   [ ] Release web version