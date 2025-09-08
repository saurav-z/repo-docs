# KittenTTS: Realistic Text-to-Speech for Everyone 😻

**Transform text into natural-sounding speech with KittenTTS, a lightweight and high-quality open-source text-to-speech model.**

Explore the power of KittenTTS for seamless voice synthesis, designed for ease of use and exceptional performance.

**[View the KittenTTS Repository on GitHub](https://github.com/KittenML/KittenTTS)**

## Key Features of KittenTTS

*   **Ultra-Lightweight Design:** The model boasts a small size of less than 25MB, making it ideal for resource-constrained environments.
*   **CPU-Optimized Performance:** Run KittenTTS effortlessly on any device without the need for a GPU.
*   **High-Quality Voice Options:** Access several premium voice options to create engaging and realistic speech outputs.
*   **Fast Inference Speed:** Experience optimized real-time speech synthesis for immediate audio generation.

## Getting Started with KittenTTS

### Installation

Install KittenTTS with a single pip command:

```bash
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
```

### Basic Usage Example

Here's how to get started with generating speech:

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

KittenTTS is designed to be highly compatible and runs on virtually any system.

## Roadmap

*   \[x] Release a preview model
*   \[ ] Release the fully trained model weights
*   \[ ] Release mobile SDK
*   \[ ] Release web version

## Get Involved

*   **Join our Discord:** [Discord Link](https://discord.com/invite/VJ86W4SURW)
*   **For Custom Support:** [Fill out our support form](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)
*   **Email us with questions:** info@stellonlabs.com