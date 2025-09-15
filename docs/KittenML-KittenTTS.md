# Kitten TTS: High-Quality Text-to-Speech in a Tiny Package

**Kitten TTS offers realistic, high-quality text-to-speech synthesis in an incredibly lightweight package, making it perfect for deployment on any device.** ( [Original Repository](https://github.com/KittenML/KittenTTS) )

## Key Features

*   **Ultra-Lightweight:** The model size is less than 25MB, enabling easy deployment.
*   **CPU-Optimized:** Runs efficiently on any device without requiring a GPU.
*   **High-Quality Voices:** Choose from several premium voice options for a natural listening experience.
*   **Fast Inference:** Optimized for real-time speech synthesis, ensuring quick generation of audio.

## Getting Started

### Installation

Install Kitten TTS using pip:

```bash
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
```

### Basic Usage

Here's how to generate speech from text:

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

## Stay Connected

*   [Join our Discord](https://discord.com/invite/VJ86W4SURW)
*   [Request Custom Support](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
*   Email us: info@stellonlabs.com

## Future Development

*   [x] Release a preview model
*   [ ] Release the fully trained model weights
*   [ ] Release mobile SDK
*   [ ] Release web version