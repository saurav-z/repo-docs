# KittenTTS: The Lightweight, High-Quality Text-to-Speech Model (😻)

**Bring your text to life with KittenTTS, an open-source, ultra-lightweight text-to-speech model that delivers realistic voice synthesis without the need for a GPU.**  ([Original Repository](https://github.com/KittenML/KittenTTS))

## Key Features:

*   **Ultra-Lightweight:** Model size under 25MB, perfect for deployment on resource-constrained devices.
*   **CPU-Optimized:**  Runs seamlessly on any device without requiring a GPU.
*   **High-Quality Voices:**  Choose from several premium voice options to match your needs.
*   **Fast Inference:** Optimized for real-time speech synthesis, enabling quick audio generation.

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

KittenTTS is designed to work on virtually any system.

##  Future Development

*   [x] Release a preview model
*   [ ] Release the fully trained model weights
*   [ ] Release mobile SDK
*   [ ] Release web version

## Get Involved

*   **Join the Community:** [Join our Discord](https://discord.com/invite/VJ86W4SURW)
*   **Custom Support:** [Fill out this form](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
*   **Contact Us:** Email the creators with any questions: info@stellonlabs.com