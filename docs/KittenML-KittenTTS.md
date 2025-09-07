# Kitten TTS: Realistic Text-to-Speech for Everyone

**Experience high-quality, realistic voice synthesis with Kitten TTS, a lightweight and accessible open-source text-to-speech model.** For more details, visit the original repository: [KittenML/KittenTTS](https://github.com/KittenML/KittenTTS)

## Key Features

*   **Ultra-Lightweight:** Model size under 25MB for easy deployment.
*   **CPU-Optimized:** Runs efficiently on any device without requiring a GPU.
*   **High-Quality Voices:** Access to a range of premium voice options.
*   **Fast Inference:** Optimized for real-time speech synthesis.

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

# Available voices: ['expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f', 'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f']

# Save the audio
import soundfile as sf
sf.write('output.wav', audio, 24000)
```

## System Requirements

Kitten TTS is designed to work on virtually any system.

## Future Development

*   Release a preview model
*   Release the fully trained model weights
*   Release mobile SDK
*   Release web version