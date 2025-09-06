# Kitten TTS: The Lightweight, High-Quality Text-to-Speech Model 🎤

**Generate realistic, expressive speech with Kitten TTS, a compact and efficient text-to-speech model designed for ease of use.** (Learn more at the [original repository](https://github.com/KittenML/KittenTTS).)

## Key Features of Kitten TTS

*   **Ultra-Lightweight Design:** The model size is less than 25MB, perfect for deployment on various devices.
*   **CPU-Optimized Performance:** Runs efficiently on any device without requiring a GPU.
*   **High-Quality Voice Options:** Access a selection of premium voice options for diverse applications.
*   **Fast Inference Speed:** Optimized for real-time speech synthesis, ensuring a seamless user experience.

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
m = KittenTTS("KittenML/kitten-tts-nano-0.2")

audio = m.generate("This high quality TTS model works without a GPU", voice='expr-voice-2-f' )

# Available voices: ['expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f', 'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f']

# Save the audio
import soundfile as sf
sf.write('output.wav', audio, 24000)
```

## System Requirements

Kitten TTS is designed to work on any device.

## Get Involved

*   [Join our Discord](https://discord.com/invite/VJ86W4SURW) for community support and updates.
*   [Contact us for custom support](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview).
*   Email us with any questions: info@stellonlabs.com

## Roadmap / Future Development

*   [x] Release a preview model
*   [ ] Release the fully trained model weights
*   [ ] Release mobile SDK
*   [ ] Release web version