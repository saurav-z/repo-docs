# Kitten TTS: Realistic Text-to-Speech for Everyone

**Kitten TTS is a cutting-edge, open-source text-to-speech model that delivers high-quality, realistic voice synthesis with a remarkably small footprint.** ([See the original repository](https://github.com/KittenML/KittenTTS))

## Key Features:

*   **Lightweight Design:** The model is under 25MB, perfect for deployment on various devices.
*   **CPU-Optimized:** Runs efficiently on any device without requiring a GPU.
*   **High-Quality Voices:** Offers a selection of premium voice options for diverse applications.
*   **Fast Inference:** Optimized for real-time speech synthesis, ensuring a smooth user experience.

## Getting Started:

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

*   **Join our Discord:** [https://discord.com/invite/VJ86W4SURW](https://discord.com/invite/VJ86W4SURW)
*   **Custom Support:** [Fill out this form](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
*   **Email:** info@stellonlabs.com