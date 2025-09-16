# Kitten TTS: Lightweight, High-Quality Text-to-Speech (TTS)

**Bring your text to life with Kitten TTS, a cutting-edge, open-source text-to-speech model that delivers realistic voices with exceptional efficiency.** [Visit the original repo](https://github.com/KittenML/KittenTTS).

## Key Features:

*   **Ultra-Lightweight:** The model size is under 25MB, perfect for resource-constrained environments.
*   **CPU-Optimized:** Runs seamlessly on any device without the need for a GPU.
*   **High-Quality Voices:** Experience a range of premium voice options for diverse applications.
*   **Fast Inference:** Enjoy real-time speech synthesis with optimized performance.

## Getting Started: Quick Installation & Usage

### Installation

Install Kitten TTS using pip:

```bash
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
```

### Basic Usage Example

Generate speech from text with just a few lines of code:

```python
from kittentts import KittenTTS
m = KittenTTS("KittenML/kitten-tts-nano-0.2")

audio = m.generate("This high quality TTS model works without a GPU", voice='expr-voice-2-f' )

# available_voices : [  'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',  'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f' ]

# Save the audio
import soundfile as sf
sf.write('output.wav', audio, 24000)
```

## System Requirements:

Kitten TTS is designed to be versatile and runs on any device without specialized hardware.

## Stay Connected and Get Support:

*   **Join our Discord:** [Discord Link](https://discord.com/invite/VJ86W4SURW)
*   **Custom Support:** [Support Form](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
*   **Email:** [info@stellonlabs.com](mailto:info@stellonlabs.com)

## Development Roadmap

*   [x] Release a preview model
*   [ ] Release the fully trained model weights
*   [ ] Release mobile SDK
*   [ ] Release web version