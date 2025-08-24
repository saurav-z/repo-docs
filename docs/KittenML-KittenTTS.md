# Kitten TTS: Realistic Text-to-Speech with Lightweight Power

**Kitten TTS** is a revolutionary, open-source text-to-speech model that delivers high-quality, realistic voice synthesis in a remarkably compact package. (See the original project on GitHub: [KittenML/KittenTTS](https://github.com/KittenML/KittenTTS)).

## Key Features

*   **Ultra-Lightweight:** The model size is less than 25MB, making it perfect for deployment on resource-constrained devices.
*   **CPU-Optimized:** Kitten TTS runs efficiently on CPUs, eliminating the need for a GPU.
*   **High-Quality Voices:** Experience a range of premium voice options for diverse applications.
*   **Fast Inference:** Optimized for real-time speech synthesis, providing a seamless user experience.

## Getting Started

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

Kitten TTS is designed to run on virtually any system.

## Future Development

*   [x] Release a preview model
*   [ ] Release the fully trained model weights
*   [ ] Release mobile SDK
*   [ ] Release web version

## Get Involved

*   Join our Discord community: [Join our discord](https://discord.com/invite/VJ86W4SURW)
*   For custom support, fill out this form: [For custom support - fill this form ](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
*   Contact the creators via email: info@stellonlabs.com