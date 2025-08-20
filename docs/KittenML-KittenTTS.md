# Kitten TTS: Ultra-Lightweight, High-Quality Text-to-Speech (TTS)

**Experience realistic voice synthesis in seconds with Kitten TTS, the open-source text-to-speech model engineered for speed and efficiency.**  [(View the original repo)](https://github.com/KittenML/KittenTTS)

Kitten TTS offers a powerful, yet compact, solution for generating natural-sounding speech. This model is ideal for developers seeking a lightweight and accessible TTS option.

**Currently in Developer Preview.**

*   [Join our Discord Community](https://discord.com/invite/VJ86W4SURW)
*   [Request Custom Support](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
*   **Contact Us:** info@stellonlabs.com

## Key Features of Kitten TTS

*   **Ultra-Lightweight:** Model size under 25MB, perfect for resource-constrained environments.
*   **CPU-Optimized:** Runs efficiently on any device without requiring a GPU.
*   **High-Quality Voices:** Choose from a selection of premium voice options.
*   **Fast Inference:** Optimized for real-time speech synthesis, providing quick results.

## Getting Started with Kitten TTS

### Installation

Install the Kitten TTS package using pip:

```bash
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
```

### Basic Usage

Here's how to generate speech using Kitten TTS:

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

Kitten TTS is designed to work virtually everywhere, requiring minimal system resources.

## Future Development

*   [x] Release a preview model
*   [ ] Release the fully trained model weights
*   [ ] Release mobile SDK
*   [ ] Release web version