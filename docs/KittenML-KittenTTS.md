# KittenTTS: Lightweight & Realistic Text-to-Speech for Everyone

**Experience high-quality, realistic voice synthesis with KittenTTS, an open-source text-to-speech model designed for fast performance and easy deployment.** ([Original Repository](https://github.com/KittenML/KittenTTS))

## Key Features of KittenTTS:

*   **Ultra-Lightweight:**  Download and deploy with ease, thanks to a model size under 25MB.
*   **CPU-Optimized:** Run KittenTTS on any device, without the need for a GPU.
*   **High-Quality Voices:** Choose from a selection of premium voice options for diverse applications.
*   **Fast Inference:** Enjoy real-time speech synthesis for a seamless user experience.

## Getting Started with KittenTTS

### Installation

Install the KittenTTS package using pip:

```bash
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
```

### Basic Usage

Here's how to generate speech from text using KittenTTS:

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

KittenTTS is designed to run on a wide range of systems.  It has no specific requirements.

## Stay Connected

*   [Join our Discord](https://discord.com/invite/VJ86W4SURW)
*   [Get Custom Support](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
*   Email: [info@stellonlabs.com](mailto:info@stellonlabs.com)

## Future Development

*   [x] Release a preview model
*   [ ] Release the fully trained model weights
*   [ ] Release mobile SDK
*   [ ] Release web version