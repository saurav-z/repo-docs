# Kitten TTS: Realistic Text-to-Speech for Everyone 😻

**Transform text into stunningly realistic speech with Kitten TTS, the open-source, lightweight text-to-speech model.**

[View the original repository on GitHub](https://github.com/KittenML/KittenTTS)

**[Join our Discord](https://discord.com/invite/VJ86W4SURW) | [Request Custom Support](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview) | [Contact Us](mailto:info@stellonlabs.com)**

## Key Features

*   **Ultra-Lightweight Design:** Model size under 25MB, perfect for resource-constrained devices.
*   **CPU-Optimized Performance:** Run Kitten TTS seamlessly on any device without the need for a GPU.
*   **High-Quality Voice Options:** Choose from a selection of premium voices to personalize your audio output.
*   **Fast and Efficient:** Experience optimized inference for real-time speech synthesis applications.

## Getting Started: Quick Installation & Usage

### Installation

```bash
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
```

### Basic Usage Example

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

Kitten TTS is designed to work on virtually any system.

## Development Roadmap

*   \[x] Release a preview model
*   \[ ] Release the fully trained model weights
*   \[ ] Release mobile SDK
*   \[ ] Release web version