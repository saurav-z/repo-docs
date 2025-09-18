# Kitten TTS: Ultra-Lightweight, High-Quality Text-to-Speech

**Tired of bulky TTS models? Kitten TTS delivers realistic, expressive speech synthesis in a tiny package.** (Original Repo: [KittenML/KittenTTS](https://github.com/KittenML/KittenTTS))

## Key Features

*   **Ultra-Lightweight:**  Model size under 25MB, perfect for resource-constrained environments.
*   **CPU-Optimized:** Runs seamlessly on any device without a GPU, making deployment easy.
*   **High-Quality Voices:** Enjoy a selection of premium voice options for diverse expression.
*   **Fast Inference:** Optimized for real-time speech synthesis, providing immediate results.

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

Kitten TTS is designed to be highly compatible and works on virtually any system.

## Roadmap

*   [x] Release a preview model
*   [ ] Release the fully trained model weights
*   [ ] Release mobile SDK
*   [ ] Release web version

## Get Involved

*   **Join the Community:**  [Join our Discord](https://discord.com/invite/VJ86W4SURW)
*   **Contact Us:**  For custom support, fill out this [form](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
*   **Email:**  info@stellonlabs.com