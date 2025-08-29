# Kitten TTS: Generate Realistic Speech on Any Device (No GPU Needed!)

Kitten TTS is a revolutionary open-source text-to-speech (TTS) model, enabling high-quality voice synthesis with exceptional efficiency.

[View the Kitten TTS Repository on GitHub](https://github.com/KittenML/KittenTTS)

## Key Features:

*   **Ultra-Lightweight:** Download and run the model with a size under 25MB.
*   **CPU-Optimized:** Runs seamlessly on any device, without the need for a GPU.
*   **High-Quality Voices:** Enjoy several premium voice options for diverse speech generation.
*   **Fast Inference:** Optimized for real-time speech synthesis and quick audio generation.

## Quick Start: Get Started with Kitten TTS

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

Kitten TTS is designed to work on virtually any device without any specific hardware requirements.

## Current Development Status & Roadmap

*   **Developer Preview:** The model is currently in developer preview.
*   **Future Development:**
    *   Release fully trained model weights
    *   Release a mobile SDK
    *   Release a web version

## Get Involved & Stay Updated

*   [Join our Discord](https://discord.com/invite/VJ86W4SURW)
*   [For custom support - fill this form ](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
*   Email the creators with any questions : info@stellonlabs.com