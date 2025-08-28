# Kitten TTS: Unleash High-Quality Text-to-Speech with Lightweight Efficiency 😻

Looking for a text-to-speech solution that delivers impressive audio quality without the need for heavy hardware? **Kitten TTS** is an open-source, ultra-lightweight text-to-speech model perfect for diverse applications.  (See the original repository [here](https://github.com/KittenML/KittenTTS)).

## Key Features

*   **Ultra-Lightweight Design:** Model size under 25MB, enabling easy deployment on a wide range of devices.
*   **CPU-Optimized:** Runs smoothly without a GPU, ensuring accessibility on nearly any hardware.
*   **High-Quality Voices:** Choose from several premium voice options for diverse speech synthesis needs.
*   **Fast Inference:** Optimized for real-time speech generation, making it ideal for interactive applications.

## Getting Started: Quick Installation & Usage

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

Kitten TTS is designed to be incredibly versatile and works on virtually any device.

## Roadmap & Future Developments

*   [x] Release a preview model
*   [ ] Release the fully trained model weights
*   [ ] Release mobile SDK
*   [ ] Release web version

## Stay Connected

*   [Join our Discord](https://discord.com/invite/VJ86W4SURW)
*   [For custom support - fill this form ](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
*   Email the creators with any questions : info@stellonlabs.com