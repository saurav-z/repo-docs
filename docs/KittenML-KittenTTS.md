# KittenTTS: Realistic Text-to-Speech, Lightweight & Open Source

**Transform text into lifelike speech with KittenTTS, a cutting-edge, open-source text-to-speech model designed for unparalleled quality and efficiency.**

**[View the original repository on GitHub](https://github.com/KittenML/KittenTTS)**

KittenTTS offers a revolutionary approach to text-to-speech, providing high-quality voice synthesis in a compact, CPU-optimized package.

## Key Features:

*   **Ultra-Lightweight:** Model size under 25MB, perfect for resource-constrained environments.
*   **CPU-Optimized:** Runs efficiently without a GPU, making it accessible on any device.
*   **High-Quality Voices:** Enjoy a selection of premium voice options for diverse applications.
*   **Fast Inference:** Optimized for real-time speech synthesis, delivering immediate results.

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

KittenTTS is designed to run virtually everywhere, requiring minimal system resources.

## Future Development Roadmap

*   [x] Release a preview model
*   [ ] Release the fully trained model weights
*   [ ] Release mobile SDK
*   [ ] Release web version

## Join the Community

*   **[Join our Discord](https://discord.com/invite/VJ86W4SURW)**
*   **For custom support - fill this form:** [Custom Support Form](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
*   **Email the creators with any questions:** info@stellonlabs.com